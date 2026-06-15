import numpy as np
import torch
import torch.nn as nn
from time import time
import matplotlib.pyplot as plt

from humancompatible.train.dual_optim import ALM, iALM

DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

xMin, xMax = -np.pi, np.pi
tMax = 5.
d_in = 2

def Nx_from_arch(width, depth, fac=2.0, d_in=2, d_out=1):
    Ntheta = (d_in + 1) * width + (depth - 1) * (width * width + width) + d_out * (width + 1)
    Ncoll_target = int(Ntheta / fac)
    Nx = int(np.sqrt(Ncoll_target))
    Nt = Nx
    return Nx, Nt, Ntheta, Ncoll_target

width, depth = 80, 4
Nx, Nt, Ntheta, Ncoll = Nx_from_arch(width=width, depth=depth, fac=10.)

dx = (xMax - xMin) / (Nx - 1)
dt = tMax / (Nt - 1)
h = max(dx, dt)

lambdas = torch.tensor([1., 1., 1.], dtype=DTYPE, device=device, requires_grad=False)
do_training = True
cheb_par = torch.tensor(0.5, dtype=DTYPE, device=device, requires_grad=False)

x = torch.linspace(xMin, xMax, Nx, dtype=DTYPE, device=device).reshape(-1, 1)
t = torch.linspace(0, tMax, Nt, dtype=DTYPE, device=device).reshape(-1, 1)
x_train = x.reshape(-1, 1)
t_train = t.reshape(-1, 1)
t_grid, x_grid = torch.meshgrid(t.flatten(), x.flatten(), indexing='ij')
inputs = torch.stack([x_grid.flatten(), t_grid.flatten()], dim=1)

save_fig = True

def u_0(x):
    return 0.2 + 0.1 * torch.cos(2 * x)

def u_0_x(x):
    return -0.2 * torch.sin(2 * x)

def periodic_boundary_conditions(model, Nbc=2000):
    x = torch.rand(Nbc, 1, device=device) * (xMax - xMin) + xMin
    t = torch.rand(Nbc, 1, device=device) * tMax

    xL = torch.full_like(x, xMin)
    xR = torch.full_like(x, xMax)
    
    uLx = model(torch.cat([xL, t], 1))
    uRx = model(torch.cat([xR, t], 1))
    
    loss = torch.mean((uLx - uRx)**2)
    return loss

def ch_density(u, u_x):
    return u**3 + u * u_x**2

def H(u, u_x, dx, density_fn=ch_density):
    f = density_fn(u, u_x)
    return torch.sum(f) * dx

def linear_loss_function(tensors, weights):
    stacked = torch.stack(tensors)
    weights = weights / torch.sum(weights)
    loss = torch.sum(weights * stacked)
    return loss, 'ls'

def chebyshev_loss_function(tensors, weights):
    stacked = torch.stack(tensors)
    loss = torch.max(weights * stacked)
    return loss, 'cs'

### MODEL ###

# @torch.compile
class PINNModel(nn.Module):
    def __init__(self, num_hidden_layers=depth, num_neurons_per_layer=width):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, num_neurons_per_layer))
            layers.append(nn.GELU())
            in_dim = num_neurons_per_layer
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def lambda_grad(epoch,
                start=1000,
                lam_max=1e-0,
                kappa=1e-3):
    epoch = float(epoch)
    return lam_max * (1.0 - np.exp(-kappa * max(epoch - start, 0.0)))


##### UNCONSTRAINED LOSS FUNCTION WITH H1 REGULARIZATION #####


# @torch.compile
def custom_loss(inputs, model, epoch):
    x, t = inputs[:, 0:1], inputs[:, 1:2]
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u_model = model(torch.cat([x, t], dim=1))
    
    u_t = torch.autograd.grad(u_model.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u_model.sum(), x, create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_xxt = torch.autograd.grad(u_xx.sum(), t, create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx.sum(), x, create_graph=True)[0]
    
    r = u_t - u_xxt + 3.0 * u_model * u_x - 2.0 * u_x * u_xx - u_model * u_xxx
    r_x = torch.autograd.grad(r.sum(), x, create_graph=True)[0]
    
    pde_loss_L2 = torch.mean(torch.square(r))
    pde_loss_grad = torch.mean(torch.square(r_x))

    lam = lambda_grad(epoch)
    pde_loss_H1 = pde_loss_L2 + lam * pde_loss_grad
    
    ic_mask = torch.abs(t) < 1e-6
    x_ic = x[ic_mask[:, 0]]
    u_ic = u_0(x_ic)
    t_ic = torch.zeros_like(x_ic)
    u_ic_pred = model(torch.cat([x_ic, t_ic], axis=1))
    data_fitting_loss_0 = torch.mean(torch.square(u_ic_pred - u_ic))
    
    data_fitting_loss_l_r = periodic_boundary_conditions(model)
    
    loss, loss_type = chebyshev_loss_function(
        [pde_loss_H1, data_fitting_loss_0, data_fitting_loss_l_r],
        lambdas
    )
    
    H_loss = H(u_model.reshape(Nt, Nx), u_x.reshape(Nt, Nx), dx)
    
    return loss, loss_type, pde_loss_L2, data_fitting_loss_0, data_fitting_loss_l_r, H_loss


#### LOSS FUNCTION WITH H1 CONSTRAINT ####

def lagrangian_loss(inputs, model, dual_opt, epoch, H0):
    x, t = inputs[:, 0:1], inputs[:, 1:2]
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u_model = model(torch.cat([x, t], dim=1))
    
    u_t = torch.autograd.grad(u_model.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u_model.sum(), x, create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_xxt = torch.autograd.grad(u_xx.sum(), t, create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx.sum(), x, create_graph=True)[0]
    
    r = u_t - u_xxt + 3.0 * u_model * u_x - 2.0 * u_x * u_xx - u_model * u_xxx
    r_x = torch.autograd.grad(r.sum(), x, create_graph=True)[0]
    
    pde_loss_L2 = torch.mean(torch.square(r))
    pde_loss_grad = torch.mean(torch.square(r_x))

    lam = lambda_grad(epoch)
    pde_loss_H1 = pde_loss_L2 + lam * pde_loss_grad
    
    ic_mask = torch.abs(t) < 1e-6
    x_ic = x[ic_mask[:, 0]]
    u_ic = u_0(x_ic)
    t_ic = torch.zeros_like(x_ic)
    u_ic_pred = model(torch.cat([x_ic, t_ic], axis=1))
    data_fitting_loss_0 = torch.mean(torch.square(u_ic_pred - u_ic))
    
    data_fitting_loss_l_r = periodic_boundary_conditions(model)
    
    loss, loss_type = chebyshev_loss_function(
        [pde_loss_H1, data_fitting_loss_0, data_fitting_loss_l_r],
        lambdas
    )

    # constraint
    Hf = H(u_model.reshape(Nt, Nx), u_x.reshape(Nt, Nx), dx)
    H_constraint = torch.abs(Hf - H0)/torch.abs(H0)

    eps = 1/(epoch+1)
    H_constraint = torch.max(H_constraint - eps, torch.zeros_like(H_constraint)).unsqueeze(0)

    loss = dual_opt.forward_update(loss, H_constraint)
    
    return loss, loss_type, pde_loss_L2, data_fitting_loss_0, data_fitting_loss_l_r, Hf


####### TRAINING LOOP #######


model = PINNModel().to(device)
epochs = 1000

optimizer = torch.optim.NAdam(model.parameters(), lr=1e-2, betas=(0.8, 0.9), eps=1e-07)
lr_schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9**(1/epochs))

losses, pde_losses, data_losses_0, bc_losses = [], [], [], []
H_losses_min, H_losses_max, H_losses_mean, H_losses_std = [], [], [], []
H_losses_abs_error, H_losses_rel_error = [], []
t0 = time()


# dual_opt = ALM(m=1, lr=5e-5, dual_range=(0.,100.), device=device, ctol=1e-3, penalty=0.)
dual_opt = iALM(m=1, beta=0.01, sigma=1.0001, gamma=1., dual_range=(0.,10.), ctol=1e-3)

H0 = H(u_0(x_grid.flatten().reshape(-1, 1)), u_0_x(x_grid.flatten().reshape(-1, 1)), dx)

for epoch in range(epochs):
    optimizer.zero_grad()
    # loss, loss_type, pde_loss, data_loss_0, bc_loss, H_loss = custom_loss(inputs, model, epoch)

    loss, loss_type, pde_loss, data_loss_0, bc_loss, H_loss = lagrangian_loss(inputs, model, dual_opt, epoch, H0)
    loss.backward()
    optimizer.step()
    
    
    if epoch % 1 == 0:
        lr_schedule.step()
    
    with torch.no_grad():

        losses.append(loss.item())
        pde_losses.append(pde_loss.item())
        data_losses_0.append(data_loss_0.item())
        bc_losses.append(bc_loss.item())

        H_loss_min = torch.min(H_loss).item()
        H_loss_max = torch.max(H_loss).item()
        H_losses_min.append(H_loss_min)
        H_losses_max.append(H_loss_max)
        H_loss_mean = torch.mean(H_loss).item()
        H_loss_std = torch.std(H_loss).item()
        H_losses_mean.append(H_loss_mean)
        H_losses_std.append(H_loss_std)
        
        # H0 = H(u_0(x_grid.flatten().reshape(-1, 1)), u_0_x(x_grid.flatten().reshape(-1, 1)), dx)
        Hf = H_loss.detach()
        H_abs_error = torch.abs(Hf - H0)
        H_losses_abs_error.append(torch.max(H_abs_error).item())
        H_rel_error = H_abs_error / (torch.abs(H0) + 1e-16)
        if isinstance(H_rel_error, torch.Tensor):
            H_rel_error = H_rel_error.item() if H_rel_error.numel() == 1 else H_rel_error.max().item()
        H_losses_rel_error.append(H_rel_error)
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6e}")

        # lambdas_values.append((trainable[-1]).numpy())
        if len(losses) > 1:
            # SoftAdaptive weights update
            num1 = np.exp(pde_losses[-1] - pde_losses[-2])
            num2 = np.exp(data_losses_0[-1] - data_losses_0[-2])
            num3 = np.exp(bc_losses[-1] - bc_losses[-2])
            den  = num1 + num2 + num3

            new_lambdas = torch.tensor([num1 / den, num2 / den, num3 / den])
            lambdas = new_lambdas
            # lambdas_values.append((lambdas).numpy())

print(f'\nComputation time: {time() - t0:.2f}s')
print(f"Loss type: {loss_type}")
print(f"Hamiltonian mean: {H_loss_mean}")
print(f"Hamiltonian std: {H_loss_std}")
print(f"Hamiltonian max: {H_loss_max}")
print(f"Hamiltonian min: {H_loss_min}")

plt.figure(figsize=(10, 6))
plt.semilogy(losses, label='Total Loss')
plt.semilogy(pde_losses, label='PDE Loss')
plt.semilogy(data_losses_0, label='Initial Conditions Loss')
plt.semilogy(bc_losses, label='Boundary Conditions Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Contributions')
plt.legend()
plt.grid()
plt.savefig('./results/ch_loss.png', dpi=300) if save_fig else None
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(H_losses_min, label='min H')
plt.plot(H_losses_max, label='max H')
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian')
plt.title('Hamiltonian over epochs')
plt.legend()
plt.grid()
plt.savefig('./results/ch_H_loss.png', dpi=300) if save_fig else None
plt.show()

H_losses_mean_arr = np.array(H_losses_mean)
H_losses_std_arr = np.array(H_losses_std)
plt.figure(figsize=(10, 6))
plt.plot(H_losses_mean_arr)
plt.fill_between(range(len(H_losses_mean_arr)), H_losses_mean_arr - H_losses_std_arr, H_losses_mean_arr + H_losses_std_arr, alpha=0.2)
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian mean')
plt.title('Hamiltonian mean over epochs with standard deviation')
plt.grid()
plt.savefig('./results/ch_H_loss_mean.png', dpi=300) if save_fig else None
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(H_losses_std_arr)
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian std')
plt.title('Hamiltonian standard deviation over epochs')
plt.grid()
plt.savefig('./results/ch_H_loss_std.png', dpi=300) if save_fig else None
plt.show()

H_losses_abs_error = np.array(H_losses_abs_error)
plt.figure(figsize=(10, 6))
plt.plot(H_losses_abs_error)
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian absolute error')
plt.title('Hamiltonian absolute error over epochs')
plt.grid()
plt.savefig('./results/ch_H_loss_abs_error.png', dpi=300) if save_fig else None
plt.show()

H_losses_rel_error = np.array(H_losses_rel_error)
plt.figure(figsize=(10, 6))
plt.plot(H_losses_rel_error)
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian relative error')
plt.title('Hamiltonian relative error over epochs')
plt.grid()
plt.savefig('./results/ch_H_loss_rel_error.png', dpi=300) if save_fig else None
plt.show()

N = 600
tspace = torch.linspace(0, tMax, N + 1, dtype=DTYPE, device=device)
xspace = torch.linspace(xMin, xMax, N + 1, dtype=DTYPE, device=device)
T_grid, X_grid = torch.meshgrid(tspace, xspace, indexing='ij')
XTgrid = torch.stack([X_grid.flatten(), T_grid.flatten()], dim=1)

with torch.no_grad():
    u_pred = model(XTgrid)
U = u_pred.reshape(N+1, N+1)

X_np = X_grid.cpu().numpy()
T_np = T_grid.cpu().numpy()
U_np = U.cpu().numpy()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_np, T_np, U_np, cmap='viridis')
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$u(x,t)$')
ax.set_title('Camassa-Holm equation')
plt.savefig('./results/ch_solution.png', dpi=300) if save_fig else None
plt.show()


import pandas as pd
df = pd.DataFrame()

df['total_loss'] = losses
df['pde_loss'] = pde_losses
df['data_fitting_loss_0'] = data_losses_0
df['data_fitting_loss_l_r'] = bc_losses
df['H_loss_min'] = H_losses_min
df['H_loss_max'] = H_losses_max
df['H_loss_mean'] = H_losses_mean
df['H_loss_std'] = H_losses_std
df['H_loss_abs_error'] = H_losses_abs_error
df['H_loss_rel_error'] = H_losses_rel_error
# df['cheb_par'] = cheb_par_values

df.to_csv('./results/camassa/torch_training_history.csv', index=False)