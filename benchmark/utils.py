from humancompatible.train.dual_optim import ALM, PBM, MoreauEnvelope
import torch
import time
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.nn import Sequential


# define the network
class ConvNet(nn.Module):
    def __init__(self, _=None, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def create_resnet():
    import torchvision
    return torchvision.models.resnet18(pretrained=False)


def create_conv_model():
    return ConvNet()

def create_model(input_shape, latent_size1=64, latent_size2=32):
    model = Sequential(
        torch.nn.Linear(input_shape, latent_size1),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size1, latent_size2),
        torch.nn.ReLU(),
        torch.nn.Linear(latent_size2, 1),
    )

    return model

def run_grid(
        m,
        primal_opt,
        dual_opt,
        param_grid,
        n_epochs,
        constraint_fn,
        constraint_bound,
        dataloader,
        data_train,
        data_val,
        mode: str,
        verbose: bool,
        constraints_to_eq: bool = False,
        use_slack: bool = False,
        constraint_tol: float = 0.,
        fuse_loss_constraint: bool = False,
        save_models = False,
        model_gen = None,
        model_kwargs = None,
        device = 'cpu',
        criterion = None
    ):
    train_logs = []
    val_logs = []
    models = []
    if mode not in ['torch', 'hc', 'sw']:
        raise ValueError(f"Expected`mode`to be one of (torch, hc, sw), got {mode}")
    
    for i, param_set in enumerate(param_grid):
        print(f"starting {i+1}/{len(param_grid)}: {param_set}")

        model, train_history, val_history = run_train(
            m=m,
            primal_opt=primal_opt,
            dual_opt=dual_opt,
            param_set=param_set,
            data_train=data_train,
            dataloader=dataloader,
            data_val=data_val,
            n_epochs=n_epochs,
            c_fn=constraint_fn,
            constraint_bound=constraint_bound,
            mode=mode,
            verbose=verbose,
            constraints_to_eq=constraints_to_eq,
            use_slack=use_slack,
            constraint_tol=constraint_tol,
            fuse_loss_constraint=fuse_loss_constraint,
            model_gen=model_gen,
            model_kwargs=model_kwargs,
            device=device,
            criterion=criterion
        )
        train_logs.append(train_history)
        val_logs.append(val_history)
        if save_models:
            models.append(model)
        else:
            del model
            models.append(None)

    return models, train_logs, val_logs 


def run_train(
    m,
    primal_opt,
    dual_opt,
    param_set,
    data_train,
    dataloader,
    data_val,
    n_epochs,
    c_fn,
    constraint_bound,
    mode='hc',
    verbose=False,
    constraints_to_eq=False,
    use_slack=False,
    fuse_loss_constraint=False,
    reg_penalty=None,
    model_gen = None,
    model_kwargs = None,
    device = 'cpu',
    constraint_tol: float = 0.,
    criterion = None,
):
    
    model = model_gen(**model_kwargs)

    primal_params = {k.removeprefix('primal__'): v for k, v in param_set.items() if k.startswith('primal__')}
    dual_params = {k.removeprefix('dual__'): v for k, v in param_set.items() if k.startswith('dual__')}
    moreau_params = {k.removeprefix('moreau__'): v for k, v in param_set.items() if k.startswith('moreau__')}
    # set up primal optimizer
    primal_optimizer = MoreauEnvelope(
        primal_opt(model.parameters(), **primal_params), **moreau_params
    ) if mode == 'hc' else primal_opt(model.parameters(), **primal_params)
    # set up slack variables if needed
    if use_slack:
        slack_vars = torch.zeros(m, requires_grad=True)
        primal_optimizer.add_param_group(param_group={"params": slack_vars, "name": "slack"})
    else:
        slack_vars = None

    if dual_opt is not None:
        if mode == 'sw':
            dual_optimizer = dual_opt(model.parameters(), **dual_params)
        else:
            dual_optimizer = dual_opt(m=m, **dual_params)
    else:
        dual_optimizer = None

    bounds = torch.tensor([constraint_bound]*m).to(device)

    # torch.sum(model(data_train[0][0].unsqueeze(0))).backward()
    # model.zero_grad()

    if mode == 'hc':
        history = train_loop_primal_dual(
            model=model,
            train_dataloader=dataloader,
            val_data=data_val,
            loss_fn=criterion,
            constraint_fn=c_fn,
            constraint_bounds=bounds,
            constraint_options={'constraints_to_eq': constraints_to_eq, "fuse_loss_constraint": fuse_loss_constraint},
            optimizer=primal_optimizer,
            dual_optimizer=dual_optimizer,
            num_epochs=n_epochs,
            device=device,
            time_constraint_computation=True,
            mode='pbm' if isinstance(dual_opt, PBM) else 'alm',
            slack_vars=slack_vars
        )
    elif mode == 'sw':
        history = train_loop_sw(
            model=model,
            train_dataloader=dataloader,
            val_data=data_val,
            loss_fn=criterion,
            constraint_fn=c_fn,
            constraint_bounds=bounds,
            constraint_options={'constraints_to_eq': constraints_to_eq, "fuse_loss_constraint": fuse_loss_constraint},
            optimizer=primal_optimizer,
            dual_optimizer=dual_optimizer,
            num_epochs=n_epochs,
            device=device,
            time_constraint_computation=True,
            constraint_tol=constraint_tol
        )
    elif mode == 'torch':
        history = train_loop_adam(
            model=model,
            train_dataloader=dataloader,
            val_data=data_val,
            loss_fn=criterion,
            constraint_fn=c_fn,
            constraint_bounds=bounds,
            constraint_options={'constraints_to_eq': constraints_to_eq, "fuse_loss_constraint": fuse_loss_constraint},
            optimizer=primal_optimizer,
            dual_optimizer=dual_optimizer,
            num_epochs=n_epochs,
            device=device,
            time_constraint_computation=True,
        )


    return history



def train_loop_sw(
    model,
    train_dataloader,
    val_data,
    loss_fn,
    constraint_fn,
    constraint_bounds,
    constraint_options,
    optimizer,
    dual_optimizer,
    num_epochs=100,
    device="cpu",
    time_constraint_computation=True,
    constraint_tol: float = 1e-3
):
    model.to(device)
    history_train = []
    history_val = []
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_constraints": [],
        "val_constraints": [],
        "train_time": [],
        "constraint_time": [] if time_constraint_computation else None
    }

    if constraint_options['constraints_to_eq']:
        raise ValueError("Switching subgradient method should not be used with equality constraints.")

    for epoch in range(num_epochs+1):

        epoch_losses = []
        epoch_constraints = []
        epoch_constraint_time = 0.0
        
        if epoch == 0:
            model.eval()
            for batch_features, batch_sens_attrs, batch_labels in train_dataloader:
                batch_features = batch_features.to(device)
                batch_sens_attrs = batch_sens_attrs.to(device)
                batch_labels = batch_labels.to(device)
                batch_out = model(batch_features)
                loss = loss_fn(batch_out, batch_labels)
                constraints, constraints_bounded_eq = calc_constraints(model, constraint_fn, constraint_bounds, constraint_options, None, batch_sens_attrs, batch_labels, batch_out, loss)
                if loss.dim() > 0:  # If loss is not aggregated
                    loss = loss.mean()
                train_start_time = time.perf_counter()
                epoch_losses.append(loss.detach().cpu().numpy().item())
                epoch_constraints.append(constraints.detach().cpu().numpy())
        else:
            # Training phase
            model.train()
            # Start training timer
            train_start_time = time.perf_counter()
            
            for batch_features, batch_sens_attrs, batch_labels in train_dataloader:
                batch_features = batch_features.to(device)
                batch_sens_attrs = batch_sens_attrs.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                batch_out = model(batch_features)
                
                # Compute constraints
                # if not timing constraints, we just log the time and then subtract it at the end.
                constraint_start = time.perf_counter()
                constraints, constraints_bounded_eq = calc_constraints(model, constraint_fn, constraint_bounds, constraint_options, None, batch_sens_attrs, batch_labels, batch_out, None)
                epoch_constraint_time += time.perf_counter() - constraint_start
                
                max_constraint = max(constraints_bounded_eq)

                # constraint step if violated
                if max_constraint > constraint_tol:
                    max_constraint.backward()
                    dual_optimizer.step()
                    loss = loss_fn(batch_out, batch_labels)
                    if loss.dim() > 0:
                        loss = loss.mean()
                else:
                    loss = loss_fn(batch_out, batch_labels)
                    if loss.dim() > 0:
                        loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                

                epoch_losses.append(loss.detach().cpu().numpy().item())
                epoch_constraints.append(constraints.detach().cpu().numpy())

        # Stop training timer
        train_end_time = time.perf_counter()
        train_time = train_end_time - train_start_time
        
        # Subtract constraint time if requested
        if not time_constraint_computation:
            train_time -= epoch_constraint_time

        eval_dict = {
            "epoch": epoch,
            "time": train_time,
            "loss": np.mean(epoch_losses)
        } | { f"c_{j}": c for j, c in enumerate(np.mean(epoch_constraints, axis=0)) }

        history_train.append(eval_dict)
        
        # Validation phase
        model.eval()
        val_loss, val_constraints = validate_model(model, val_data, loss_fn, constraint_fn, device)
        
        eval_dict = {
            "epoch": epoch,
            "time": train_time,
            "loss": val_loss
        } | {
            f"c_{j}": c for j, c in enumerate(val_constraints)
        }

        history_val.append(eval_dict)

    
    return model, history_train, history_val
    

def train_loop_primal_dual(
    model,
    train_dataloader,
    val_data,
    loss_fn,
    constraint_fn,
    constraint_bounds,
    constraint_options,
    optimizer,
    dual_optimizer=None,
    num_epochs=100,
    device="cpu",
    time_constraint_computation=True,
    mode: str = 'alm',
    slack_vars: torch.Tensor = None,
):
    model.to(device)
    history_train = []
    history_val = []
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_constraints": [],
        "val_constraints": [],
        "train_time": [],
        "constraint_time": [] if time_constraint_computation else None
    }

    if dual_optimizer is None:
        raise ValueError("Dual optimizer must be provided for hc-train optimization mode.")
    if mode =='pbm' and constraint_options['constraints_to_eq']:
        raise ValueError("SPBM method should not be used with equality constraints.")
    
    for epoch in range(num_epochs+1):
        # Training phase
        model.train()
        epoch_losses = []
        epoch_constraints = []
        epoch_constraint_time = 0.0
        
        if epoch == 0:
            model.eval()
            for batch_features, batch_sens_attrs, batch_labels in train_dataloader:
                batch_features = batch_features.to(device)
                batch_sens_attrs = batch_sens_attrs.to(device)
                batch_labels = batch_labels.to(device)
                batch_out = model(batch_features)
                loss = loss_fn(batch_out, batch_labels)
                # print(loss)
                constraints, constraints_bounded_eq = calc_constraints(model, constraint_fn, constraint_bounds, constraint_options, None, batch_sens_attrs, batch_labels, batch_out, loss)
                # print(constraints)
                # print(constraints_bounded_eq)
                if loss.dim() > 0:  # If loss is not aggregated
                    loss = loss.mean()
                train_start_time = time.perf_counter()
                epoch_losses.append(loss.detach().cpu().numpy().item())
                epoch_constraints.append(constraints.detach().cpu().numpy())
            model.train()
        else:
            # Start training timer
            train_start_time = time.perf_counter()
            for batch_features, batch_sens_attrs, batch_labels in train_dataloader:
                if slack_vars is not None:
                    with torch.no_grad():
                        for s in slack_vars:
                            if s < 0:
                                s.zero_()
                batch_features = batch_features.to(device)
                batch_sens_attrs = batch_sens_attrs.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                # Forward pass
                batch_out = model(batch_features)
                # Compute loss
                loss = loss_fn(batch_out, batch_labels)
                # Compute constraints
                # if not timing constraints, we just log the time and then subtract it at the end.
                constraint_start = time.perf_counter()
                # print(loss)
                constraints, constraints_bounded_eq = calc_constraints(model, constraint_fn, constraint_bounds, constraint_options, slack_vars, batch_sens_attrs, batch_labels, batch_out, loss)
                epoch_constraint_time += time.perf_counter() - constraint_start
                
                if loss.dim() > 0:  # If loss is not aggregated
                    loss = loss.mean()
                    
                lgr = dual_optimizer.forward_update(loss, constraints_bounded_eq)
                lgr.backward()
                optimizer.step()
                
                epoch_losses.append(loss.detach().cpu().numpy().item())
                epoch_constraints.append(constraints.detach().cpu().numpy())

        # Stop training timer
        train_end_time = time.perf_counter()
        train_time = train_end_time - train_start_time
        
        # Subtract constraint time if requested
        if not time_constraint_computation:
            train_time -= epoch_constraint_time

        eval_dict = {
            "epoch": epoch,
            "time": train_time,
            "loss": np.mean(epoch_losses)
        } | { f"c_{j}": c for j, c in enumerate(np.mean(epoch_constraints, axis=0)) }

        history_train.append(eval_dict)
        
        # Validation phase
        model.eval()
        val_loss, val_constraints = validate_model(model, val_data, loss_fn, constraint_fn, device)
        
        eval_dict = {
            "epoch": epoch,
            "time": train_time,
            "loss": val_loss
        } | {
            f"c_{j}": c for j, c in enumerate(val_constraints)
        }

        history_val.append(eval_dict)
    
    return model, history_train, history_val


def train_loop_adam(
    model,
    train_dataloader,
    val_data,
    loss_fn,
    constraint_fn,
    constraint_bounds,
    constraint_options,
    optimizer,
    dual_optimizer=None,
    num_epochs=100,
    device="cpu",
    time_constraint_computation=False,
    slack_vars: torch.Tensor = None,
    constraint_tol: float = None,
    reg_penalty: float = None,
):
    model.to(device)
    history_train = []
    history_val = []
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_constraints": [],
        "val_constraints": [],
        "train_time": [],
        "constraint_time": [] if time_constraint_computation else None
    }

    for epoch in range(num_epochs+1):
        # Training phase
        model.train()
        epoch_losses = []
        epoch_constraints = []
        epoch_constraint_time = 0.0
        
        if epoch == 0:
            model.eval()
            for batch_features, batch_sens_attrs, batch_labels in train_dataloader:
                batch_features = batch_features.to(device)
                batch_sens_attrs = batch_sens_attrs.to(device)
                batch_labels = batch_labels.to(device)
                batch_out = model(batch_features)
                loss = loss_fn(batch_out, batch_labels)
                constraints, constraints_bounded_eq = calc_constraints(model, constraint_fn, constraint_bounds, constraint_options, None, batch_sens_attrs, batch_labels, batch_out, loss)
                if loss.dim() > 0:  # If loss is not aggregated
                    loss = loss.mean()
                train_start_time = time.perf_counter()
                epoch_losses.append(loss.detach().cpu().numpy().item())
                epoch_constraints.append(constraints.detach().cpu().numpy())
            model.train()
        else:
            # Start training timer
            train_start_time = time.perf_counter()
            
            for batch_features, batch_sens_attrs, batch_labels in train_dataloader:
                batch_features = batch_features.to(device)
                batch_sens_attrs = batch_sens_attrs.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                batch_out = model(batch_features)
                
                # Compute loss
                loss = loss_fn(batch_out, batch_labels)
                
                # Compute constraints
                constraint_start = time.perf_counter()
                constraints, constraints_bounded_eq = calc_constraints(model, constraint_fn, constraint_bounds, constraint_options, slack_vars, batch_sens_attrs, batch_labels, batch_out, loss)
                if not time_constraint_computation: # time constraint separately to subtract it from total time later
                    epoch_constraint_time += time.perf_counter() - constraint_start
                
                if loss.dim() > 0:  # If loss is not aggregated
                    loss = loss.mean()

                if reg_penalty is not None:
                    loss_penalized = loss + reg_penalty * constraints_bounded_eq.mean()
                else:
                    loss_penalized = loss

                # Backward pass
                loss_penalized.backward()
                optimizer.step()
            
                epoch_losses.append(loss.detach().cpu().numpy().item())
                epoch_constraints.append(constraints.detach().cpu().numpy())

        # Stop training timer
        train_end_time = time.perf_counter()
        train_time = train_end_time - train_start_time
        
        # Subtract constraint time if requested
        if not time_constraint_computation:
            train_time -= epoch_constraint_time

        eval_dict = {
            "epoch": epoch,
            "time": train_time,
            "loss": np.mean(epoch_losses)
        } | { f"c_{j}": c for j, c in enumerate(np.mean(epoch_constraints, axis=0)) }

        history_train.append(eval_dict)
        
        # Validation phase
        model.eval()
        val_loss, val_constraints = validate_model(model, val_data, loss_fn, constraint_fn, device)
        
        eval_dict = {
            "epoch": epoch,
            "time": train_time,
            "loss": val_loss
        } | {
            f"c_{j}": c for j, c in enumerate(val_constraints)
        }

        history_val.append(eval_dict)
        
    
    return model, history_train, history_val

def validate_model(model, val_data, loss_fn, constraint_fn, device):
    val_losses = []
    val_constraints_list = []
    with torch.no_grad():
        if isinstance(val_data, torch.utils.data.DataLoader):
            # Use dataloader batches
            for batch_features, batch_sens_attrs, batch_labels in val_data:
                batch_features = batch_features.to(device)
                batch_sens_attrs = batch_sens_attrs.to(device)
                batch_labels = batch_labels.to(device)
                    
                val_out = model(batch_features)
                val_loss = loss_fn(val_out, batch_labels)
                if val_loss.dim() > 0:  # If loss is not aggregated
                    val_loss = val_loss.mean()
                    
                val_constraints = constraint_fn(model, val_out, batch_sens_attrs, batch_labels)
                # val_constraints -= constraint_bounds
                val_losses.append(val_loss.detach().cpu().numpy().item())
                val_constraints_list.append(val_constraints.detach().cpu().numpy())
        else:
            # Use full validation dataset
            val_features, val_sens_attrs, val_labels = val_data
            val_features = val_features.to(device) if torch.is_tensor(val_features) else val_features
            val_sens_attrs = val_sens_attrs.to(device) if torch.is_tensor(val_sens_attrs) else val_sens_attrs
            val_labels = val_labels.to(device) if torch.is_tensor(val_labels) else val_labels
                
            val_out = model(val_features)
            val_loss = loss_fn(val_out, val_labels)
            if val_loss.dim() > 0:  # If loss is not aggregated
                val_loss = val_loss.mean()
                
            val_constraints = constraint_fn(model, val_out, val_sens_attrs, val_labels)
            val_losses.append(val_loss.detach().cpu().numpy().item())
            val_constraints_list.append(val_constraints.detach().cpu().numpy())

    val_losses = np.mean(val_losses)
    val_constraints_list = np.mean(val_constraints_list, axis=0)
    return val_losses, val_constraints_list


def calc_constraints(model, constraint_fn, constraint_bounds, constraint_options, slack_vars, batch_sens, batch_labels, batch_out, loss=None):
    if constraint_options.get('fuse_loss_constraint', False):
        constraints = constraint_fn(model, batch_out, batch_sens, batch_labels, loss=loss)
    else:
        constraints = constraint_fn(model, batch_out, batch_sens, batch_labels)
    # transform to equality if needed
    if constraint_options.get('constraints_to_eq', False) and slack_vars is not None:
        constraints_bounded_eq = (constraints - constraint_bounds + slack_vars)
    elif constraint_options.get('constraints_to_eq', False):
        constraints_bounded_eq = torch.max(constraints - constraint_bounds, torch.zeros_like(constraints))
    else:
        constraints_bounded_eq = (constraints - constraint_bounds)
    return constraints, constraints_bounded_eq

