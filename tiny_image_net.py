import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
from pathlib import Path
from tinyimagenet import TinyImageNet
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn
from fairret.statistic import PositiveRate, TruePositiveRate, FalsePositiveRate, PositivePredictiveValue, FalseOmissionRate
from fairret.loss import NormLoss
from humancompatible.train.fairness.utils import BalancedBatchSampler
import torch
from tqdm import tqdm
import os
from typing import Callable, Any, Dict
from dataclasses import dataclass
import torch
import fairret
from humancompatible.train.dual_optim import ALM, MoreauEnvelope, PBM


def dataset_to_tensors(dataset, batch_size=512, num_workers=8):
    """Fast parallel loading of an entire dataset into tensors."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    all_X, all_targets = [], []
    for X_batch, target_batch in tqdm(loader, desc="Loading dataset into tensors", total=len(loader)):
        all_X.append(X_batch)
        all_targets.append(target_batch)
    return torch.cat(all_X, dim=0), torch.cat(all_targets, dim=0)
 
 
def load_or_cache(dataset, cache_path, batch_size=512, num_workers=8):
    """Load tensors from cache if available, otherwise build and save."""
    if os.path.exists(cache_path):
        print(f"Loading cache from {cache_path}...")
        data = torch.load(cache_path, weights_only=True)
        return data["X"], data["targets"]
 
    print(f"Building cache → {cache_path} (one-time cost)...")
    X, targets = dataset_to_tensors(dataset, batch_size=batch_size, num_workers=num_workers)
    torch.save({"X": X, "targets": targets}, cache_path)
    print(f"Cache saved ({X.nbytes / 1e9:.2f} GB)")
    return X, targets
 
 

def train_tinyimagenet():

    # define batch size here
    batch_size = 1200
    
    # define the path here 
    dataset_path="~/.torchvision/tinyimagenet/"


    # define transforms function
    normalize_transform = T.Compose([ T.ToTensor(),
                                    T.Normalize(mean=TinyImageNet.mean,
                                std=TinyImageNet.std),
                                # Converting cropped images to tensors
    ])
    train_transform = T.Compose([ T.Resize(256), # Resize images to 256 x 256
                    T.CenterCrop(224), # Center crop image
                    T.RandomHorizontalFlip(),
                    normalize_transform

                    ])

    # --- Load datasets ---
    train = TinyImageNet(Path(dataset_path), split="train", transform=train_transform, imagenet_idx=False)
    val_full = TinyImageNet(Path(dataset_path), split="val", transform=normalize_transform, imagenet_idx=False)

    print(f"Dataset has {len(train.classes)} classes. Sample classes: {train.classes[:5]}")
    
    # --- Cache and split val into val/test ---
    X_val_full, targets_val_full = load_or_cache(val_full, cache_path="./data/cache_val.pt")

    n = len(X_val_full)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    split = n // 2
    val_idx, test_idx = idx[:split], idx[split:]

    raw_splits = {
        "train": load_or_cache(train, cache_path="./data/cache_train.pt"),
        "val":   (X_val_full[val_idx],  targets_val_full[val_idx]),
        "test":  (X_val_full[test_idx], targets_val_full[test_idx]),
    }

    # --- Build loaders ---
    loaders = {}
    for name, (X, targets) in raw_splits.items():
        print(f"\nDataset: {name} | Size: {len(X)}")
        print(f"  X: {X.shape}, targets: {targets.shape}")

        groups_onehot = torch.eye(200)[targets]
        dataset_torch = torch.utils.data.TensorDataset(X, groups_onehot, targets)

        group_counts = torch.bincount(targets, minlength=200)
        print("Samples per group:", group_counts)

        sampler = BalancedBatchSampler(
            group_onehot=groups_onehot, batch_size=batch_size, drop_last=True
        )
        loaders[name] = torch.utils.data.DataLoader(
            dataset_torch, batch_size=batch_size, shuffle=True, num_workers=4
        )
        loaders[name + "_balanced"] = torch.utils.data.DataLoader(
            dataset_torch, batch_sampler=sampler, num_workers=4
        )
        print(f"  Loaders created: '{name}' and '{name}_balanced'")

    # create fair dataloaders

    # ----- Build model, criterion, optimizer -----
    device = torch.device("cuda")
    epochs = 5
    loader_name = "val_balanced"


    # ----- Unconstrained Optimization Adam -----
    constraint_type = LossPairwise(loss=nn.CrossEntropyLoss(reduction='none'))
    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')  # Unaggregated loss for fairness constraints
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "max_constr": []}
 
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, max_constr = run_epoch(model, loaders[loader_name], 
                                                      criterion, optimizer, device, 
                                                      train=True)
        val_loss,   val_acc, max_constr   = run_epoch(model, loaders["val_balanced"], 
                                                      criterion, optimizer, device, 
                                                      train=False)
        # scheduler.step()
 
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["max_constr"].append(max_constr)
        print(f"Epoch {epoch:>3}/{epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}"
                f" | max constraint {max_constr:.4f}")  
 

    # ----- SPMB Optimization -----
    constraint_type = LossPairwise(loss=nn.CrossEntropyLoss(reduction='none'))
    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')  # Unaggregated loss for fairness constraints
    
    # Define data and optimizers
    optimizer = MoreauEnvelope(torch.optim.Adam(model.parameters(), lr=0.002), mu=2.0)
    
    dual = PBM(
        m=39800,
        # penalty_update='dimin',
        # penalty_update='dimin_adapt',
        penalty_update='const',
        pbf = 'quadratic_reciprocal',
        gamma=0.95,
        init_duals=0.00001,
        init_penalties=1.,
        penalty_range=(0.5, 1.),
        penalty_mult=0.99,
        dual_range=(0.000001, 100.),
        delta=1.0,
        device=device
    )
 
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "max_constr": []}
 
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, max_constr = run_epoch(model, loaders[loader_name],  
                                                criterion, optimizer, device, 
                                                train=True, dual=dual)
        val_loss,   val_acc, max_constr   = run_epoch(model, loaders["val_balanced"], 
                                                criterion, optimizer, device, 
                                                train=False, dual=dual)
        # scheduler.step()
        
        # print numer of duals smaller than 1e-5 and larger than 100
        print('small duals', (dual.duals <= 1e-5).sum().item())
        print('large duals', (dual.duals >= 100.).sum().item())
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["max_constr"].append(max_constr)
        print(f"Epoch {epoch:>3}/{epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}"
                f" | max constraint {max_constr:.4f}")  
        
    
    # ----- SSLALM Optimization -----
    # constraint_type = LossPairwise(loss=nn.CrossEntropyLoss(reduction='none'))
    # model     = build_model().to(device)
    # criterion = nn.CrossEntropyLoss(reduction='none')  # Unaggregated loss for fairness constraints
    
    # # Define data and optimizers
    # optimizer = MoreauEnvelope(torch.optim.Adam(model.parameters(), lr=0.005), mu=2.0)
    
    # dual = ALM(
    #     m=39800,
    #     lr=0.1,
    #     momentum=0.5,
    #     # penalty_update='dimin',
    #     device=device   
    # )  
 
    # history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "max_constr": []}
 
    # for epoch in range(1, epochs + 1):
    #     train_loss, train_acc, max_constr = run_epoch(model, loaders[loader_name],  
    #                                             criterion, optimizer, device, 
    #                                             train=True, dual=dual)
    #     val_loss,   val_acc, max_constr   = run_epoch(model, loaders["val_balanced"], 
    #                                             criterion, optimizer, device, 
    #                                             train=False, dual=dual)
    #     # scheduler.step()
 
    #     history["train_loss"].append(train_loss)
    #     history["train_acc"].append(train_acc)
    #     history["val_loss"].append(val_loss)
    #     history["val_acc"].append(val_acc)
    #     history["max_constr"].append(max_constr)
    #     print(f"Epoch {epoch:>3}/{epochs} | "
    #           f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
    #           f"val loss {val_loss:.4f} acc {val_acc:.3f}"
    #             f" | max constraint {max_constr:.4f}")  


def build_model(num_classes=200):
    """EfficientNet-B0 from scratch (no pretrained weights)."""
    model = models.efficientnet_b0(weights=None)
    # Replace classifier head for 200-class TinyImageNet
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

 
def run_epoch(model, loader, criterion, optimizer, device, train=True, dual=None):
    model.train() if train else model.eval()
    total_loss, correct, total, total_constr = 0.0, 0, 0, 0.0
    constraint_type = LossPairwise(loss=nn.CrossEntropyLoss(reduction='none'))
    threshold = 0.1  # Example threshold for constraint violation
 
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, sens, y in tqdm(loader, desc="train" if train else "eval", leave=False):
            x, sens, y = x.to(device), sens.to(device), y.to(device)
 
            if train:
                optimizer.zero_grad()

            if dual is None:
                pred = model(x)
                loss = criterion(pred, y)
                # calculate the constraints
                constraints = constraint_type.compute_constraints(None, None, sens, None, loss=loss)
                constraints = constraints - threshold
                max_constr = constraints.max().item()

                if train:
                    loss.mean().backward()  # Aggregate loss for backward pass
                    optimizer.step()


            elif dual is not None:
                pred = model(x)
                loss = criterion(pred, y)
                constraints = constraint_type.compute_constraints(None, None, sens, None, loss=loss)
                constraints = constraints - threshold
                max_constr = constraints.max().item()

                # compute the lagrangian value
                lagrangian = dual.forward_update(loss.mean(), constraints)

                if train:
                    lagrangian.backward()
                    optimizer.step()
                    optimizer.zero_grad()
 
            total_loss += loss.mean().item() * x.size(0)
            correct    += (pred.argmax(1) == y).sum().item()
            total      += x.size(0)
            total_constr += max_constr

    return total_loss / total, correct / total, total_constr / len(loader)
 






def positive_rate_per_group(out_batch, batch_sens, prob_f=torch.nn.functional.sigmoid):
    """
    Calculates the positive rate vector based on the given outputs of the model for the given groups. 
    
    """
    if prob_f is None: 
        preds = out_batch
    else: 
        preds = prob_f( out_batch )
    pr = PositiveRate()
    probs_per_group = pr(preds, batch_sens)

    return probs_per_group

def posrate_per_group(model, out, batch_sens, batch_labels):
    pos_rate_pergroup = positive_rate_per_group(out, batch_sens)
    constraints = ((pos_rate_pergroup.unsqueeze(1) - pos_rate_pergroup.unsqueeze(0)).to(torch.float))
    mask = ~torch.eye(batch_sens.shape[-1], dtype=torch.bool)
    constraints = constraints[mask]

    return constraints

def posrate_fairret_constraint(model, out, batch_sens, batch_labels):
    statistic = PositiveRate()
    fair_criterion = NormLoss(statistic=statistic)
    
    return fair_criterion(out, batch_sens).unsqueeze(0)

def weight_constraint(model, out, batch_sens, batch_labels):
    norms = []
    for param in model.parameters():
        norm = torch.linalg.norm(param)
        norms.append(norm.unsqueeze(0))
    
    return torch.concat(norms)


@dataclass
class ConstraintMetadata:
    """This class is a wrapper for fairness constraints;
    it contains the function that computes the constraint
    and a function that computes the number of constraints given the number of protected groups
    (for example, if the constraint calculates a metric for each pair of groups,`m`would be`n_groups`* (`n_groups` - 1))."""
    fn: Callable[[Any, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    m_fn: Callable[[int], int]

class LossPairwise(ConstraintMetadata):
    """Wrapper class for a fairness constraint that enforces equal loss across groups.
    The constraint is computed as the pairwise difference between the losses for each group."""
    def __init__(self, loss: Callable = None, abs_diff: bool = False):
        """
        Args:
            loss (Callable): A function that computes the loss for each sample in the batch; must be **unaggregated** (i.e., reduction='none')
            If not provided, the constraint will expect the loss to be precomputed and passed as an argument to the compute_constraints function.
        """
        super().__init__(
            fn=self.compute_constraints,
            m_fn=lambda n_groups: n_groups * (n_groups - 1) if not self.abs_diff else n_groups * (n_groups - 1) // 2
        )
        self.abs_diff = abs_diff
        if self.abs_diff:
            raise NotImplementedError("abs_diff=True is not implemented yet.")
        self.loss = loss

    def compute_constraints(self, model, batch_out, batch_sens, batch_labels, loss = None):
        if loss is None:
            loss = self.loss(batch_out, batch_labels)

        per_group_losses = _get_normalized_per_group_losses(loss, batch_sens).squeeze()
        constraints = ((per_group_losses.unsqueeze(1) - per_group_losses.unsqueeze(0)))    
        mask = ~torch.eye(batch_sens.shape[-1], dtype=torch.bool)
        constraints = constraints[mask]
        return constraints

def _get_normalized_per_group_losses(loss, sens_onehot):
    return loss.unsqueeze(0) @ sens_onehot / sens_onehot.sum(dim=0)


if __name__ == "__main__":
    train_tinyimagenet()