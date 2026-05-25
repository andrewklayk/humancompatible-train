import numpy as np
import torch
import torch.nn as nn
from time import time
import matplotlib.pyplot as plt
from humancompatible.train.dual_optim import ALM

DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

type_pde = 1
if type_pde == 1:
    nu, alpha, rho = -0.022**2, -0.5, 0.
    xMin, xMax, tMax = 0., 2., 5.
elif type_pde == 2:
    nu, alpha, rho = -1., -3., 0.
    xMin, xMax, tMax = -20., 20., 100.

def Nx_from_arch(width, depth, fac=1.5, d_in=2, d_out=1):
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
# do_training = False
cheb_par = torch.tensor(0.5, dtype=DTYPE, device=device, requires_grad=False)

x = torch.linspace(xMin, xMax, Nx, dtype=DTYPE, device=device).reshape(-1, 1)
t = torch.linspace(0, tMax, Nt, dtype=DTYPE, device=device).reshape(-1, 1)
x_train = x.reshape(-1, 1)
t_train = t.reshape(-1, 1)
# t_grid, x_grid = torch.meshgrid(t.flatten(), x.flatten(), indexing='ij')
x_grid, t_grid = torch.meshgrid(x.flatten(), t.flatten(), indexing='xy')
inputs = torch.stack([x_grid.flatten(), t_grid.flatten()], dim=1)

save_fig = True

def u_0(x):
    if type_pde == 1:
        return torch.cos(np.pi * x)
    elif type_pde == 2:
        return 6. / (torch.cosh(x)**2)

def u_0_x(x):
    if type_pde == 1:
        return -np.pi * torch.sin(np.pi * x)
    elif type_pde == 2:
        return -12. * torch.sinh(x) / (torch.cosh(x)**3)

def periodic_bc(model, x, t):
    xL = torch.full_like(x, xMin)
    xR = torch.full_like(x, xMax)
    uL = model(torch.cat([xL, t], dim=1))
    uR = model(torch.cat([xR, t], dim=1))
    return torch.mean((uL - uR)**2)

def V(u):
    return alpha * (u**3) / 3 + (rho * u**2) / 2

def kdv_density(u, u_x):
    return V(u) - nu * torch.pow(u_x, 2) / 2


def H(u, u_x, dx, density_fn=kdv_density, axis=-1):
    """
    Boole’s rule (8th order) along `axis` for a uniform grid with spacing dx.
    Requires (N-1) % 4 == 0. Otherwise applies Boole on the largest valid
    prefix and trapezoid rule on the remainder.
    """
    f = density_fn(u, u_x)  # [..., N]
    n = f.shape[axis]

    # Normalize negative axis
    axis = axis % f.ndim

    def _trap_rem(rem):
        """
        rem: contiguous tail segment integrated with trapezoid rule
        """
        left = rem.narrow(-1, 0, rem.shape[-1] - 1)
        right = rem.narrow(-1, 1, rem.shape[-1] - 1)
        return torch.sum(0.5 * (left + right), dim=-1) * dx

    # Degenerate case
    if n <= 1:
        return torch.sum(f, dim=axis) * dx

    # Largest prefix satisfying (n1 - 1) % 4 == 0
    n1 = n - ((n - 1) % 4)

    # Boole constant
    c = (2.0 * dx) / 45.0

    # Build index slices
    idx_prefix = torch.arange(n1, device=f.device)

    f0 = torch.index_select(f, axis, idx_prefix[0::4])
    f1 = torch.index_select(f, axis, idx_prefix[1::4])
    f2 = torch.index_select(f, axis, idx_prefix[2::4])
    f3 = torch.index_select(f, axis, idx_prefix[3::4])
    f4 = torch.index_select(f, axis, idx_prefix[4::4])

    # Weighted Boole sum
    s = 7.0 * torch.sum(f0, dim=axis)
    s += 32.0 * torch.sum(f1, dim=axis)
    s += 12.0 * torch.sum(f2, dim=axis)
    s += 32.0 * torch.sum(f3, dim=axis)
    s += 7.0 * torch.sum(f4, dim=axis)

    boole_part = c * s

    # No remainder
    if n1 == n:
        return boole_part

    # Tail remainder
    rem_idx = torch.arange(n1 - 1, n, device=f.device)
    rem = torch.index_select(f, axis, rem_idx)

    tail = _trap_rem(rem)

    return boole_part + tail

def linear_loss_function(tensors, weights):
    stacked = torch.stack(tensors)
    weights = weights / torch.sum(weights)
    loss = torch.sum(weights * stacked)
    return loss, 'ls'

def chebyshev_loss_function(tensors, weights):
    stacked = torch.stack(tensors)
    loss = torch.max(weights * stacked)
    return loss, 'cs'

def sigmoid_centered(x):
    return 2 * torch.sigmoid(x) - 1

class PINNModel(nn.Module):
    def __init__(self, num_hidden_layers=depth, num_neurons_per_layer=width):
        super().__init__()
        layers = []
        in_dim = 2
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, num_neurons_per_layer))
            layers.append(nn.Tanh())
            in_dim = num_neurons_per_layer
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def lambda_grad(epoch,
                start=1000,
                lam_max=1e0,
                kappa=1e-3):
    epoch = torch.as_tensor(epoch, dtype=torch.float32)
    return lam_max * (
        1.0 - torch.exp(-kappa * torch.clamp(epoch - start, min=0.0))
    )


def grad_L2_fft_batch(r, L):
    """
    r : shape (Nt, Nx)
    returns : shape (Nt,)
    """
    r = r.to(torch.complex64)
    Nx = r.shape[-1]

    device = r.device

    k_pos = torch.arange(0, Nx // 2 + 1,
                         dtype=torch.float32,
                         device=device)

    k_neg = torch.arange(-Nx // 2 + 1, 0,
                         dtype=torch.float32,
                         device=device)

    k = torch.cat([k_pos, k_neg], dim=0)
    k = (2.0 * np.pi / L) * k
    k = k.to(torch.complex64)

    r_hat = torch.fft.fft(r, dim=-1)

    grad_energy = torch.sum(torch.abs(1j * k * r_hat) ** 2, dim=-1)

    dx = L / float(Nx)

    return torch.real(grad_energy) * dx


def H1_norm_fft_batch(r, L):
    """
    Compute ||r||_{H^1}^2 for each time slice.

    r : shape (Nt, Nx)
    returns : shape (Nt,)
    """
    r = r.to(torch.complex64)
    Nx = r.shape[-1]

    device = r.device

    k_pos = torch.arange(0, Nx // 2 + 1,
                         dtype=torch.float32,
                         device=device)

    k_neg = torch.arange(-Nx // 2 + 1, 0,
                         dtype=torch.float32,
                         device=device)

    k = torch.cat([k_pos, k_neg], dim=0)
    k = (2.0 * np.pi / L) * k
    k = k.to(torch.complex64)

    r_hat = torch.fft.fft(r, dim=-1)

    weight = 1.0 + torch.abs(k) ** 2

    H1_sq = torch.sum(weight * torch.abs(r_hat) ** 2, dim=-1)

    dx = L / float(Nx)

    return torch.real(H1_sq) * dx


def custom_loss(inputs, model, epoch):
    """
    Assumes the following globals/functions exist:

        alpha, rho, nu
        Nt, Nx
        dx, xMin, xMax
        lambdas

        u_0(...)
        periodic_bc(...)
        linear_loss_function(...)
        H(...)
    """

    x = inputs[:, 0:1].clone().detach().requires_grad_(True)
    t = inputs[:, 1:2].clone().detach().requires_grad_(True)

    xt = torch.cat([x, t], dim=1)

    # Forward pass
    u_model = model(xt)

    # First derivatives
    u_x = torch.autograd.grad(
        u_model,
        x,
        grad_outputs=torch.ones_like(u_model),
        create_graph=True,
        retain_graph=True,
    )[0]

    u_t = torch.autograd.grad(
        u_model,
        t,
        grad_outputs=torch.ones_like(u_model),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Second derivative
    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Third derivative
    u_xxx = torch.autograd.grad(
        u_xx,
        x,
        grad_outputs=torch.ones_like(u_xx),
        create_graph=True,
        retain_graph=True,
    )[0]

    # PDE residual
    u_squared_x = 2 * u_model * u_x

    r = (
        u_t
        - alpha * u_squared_x
        - rho * u_x
        - nu * u_xxx
    )

    # === PDE residual loss ===
    pde_loss_L2 = torch.mean(r ** 2)

    r_grid = r.reshape(Nt, Nx)

    L = xMax - xMin

    pde_loss_grad = torch.mean(
        grad_L2_fft_batch(r_grid, L)
    )

    # mesh-scaled stabilization parameter
    lam = 0.01 * (dx ** 2) * min(1.0, float(epoch) / 1000.0)

    pde_loss_H1 = pde_loss_L2 + lam * pde_loss_grad

    # === Initial condition ===
    ic_mask = torch.where(torch.abs(t) < 1e-6)[0]

    x_ic = x[ic_mask]

    u_ic = u_0(x_ic)

    t_ic = torch.zeros_like(x_ic)

    u_ic_pred = model(torch.cat([x_ic, t_ic], dim=1))

    data_fitting_loss_0 = torch.mean(
        (u_ic_pred - u_ic) ** 2
    )

    # === Periodic BC ===
    data_fitting_loss_l_r = periodic_bc(model, x, t)

    # === Aggregated loss ===
    loss, loss_type = linear_loss_function(
        [
            pde_loss_H1,
            data_fitting_loss_0,
            data_fitting_loss_l_r,
        ],
        lambdas
    )

    # === Hamiltonian (monitor only) ===
    # breakpoint()
    H_loss = H(
        u_model.reshape(Nt, Nx),
        u_x.reshape(Nt, Nx),
        dx
    )

    return (
        loss,
        loss_type,
        pde_loss_H1,
        data_fitting_loss_0,
        data_fitting_loss_l_r,
        H_loss,
    )

def lagrangian_loss(inputs, model, dual_opt):
    x, t = inputs[:, 0:1], inputs[:, 1:2]
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u_model = model(torch.cat([x, t], dim=1))
    
    u_t = torch.autograd.grad(u_model.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u_model.sum(), x, create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx.sum(), x, create_graph=True)[0]
    
    u_squared_x = 2 * u_model * u_x
    r = u_t - alpha * u_squared_x - rho * u_x - nu * u_xxx
    
    pde_loss_L2 = torch.mean(torch.square(r))
    
    # constraint: pde_loss_L2 = 0 or <= eps

    ic_mask = torch.abs(t) < 1e-6
    x_ic = x[ic_mask[:, 0]]
    u_ic = u_0(x_ic)
    t_ic = torch.zeros_like(x_ic)
    u_ic_pred = model(torch.cat([x_ic, t_ic], axis=1))
    data_fitting_loss_0 = torch.mean((u_ic_pred - u_ic) ** 2) # IC loss
    
    data_fitting_loss_l_r = periodic_bc(model, x, t) # BC loss

    # ask: what's H_loss here, and should we use it in a constraint
    H_loss = H(u_model.reshape(Nt, Nx), u_x.reshape(Nt, Nx), dx)
    
    data_fitting_loss = 0.5 * data_fitting_loss_0 + 0.5 * data_fitting_loss_l_r

    lagr = dual_opt.forward_update(data_fitting_loss, pde_loss_L2.unsqueeze(0))
    loss_type = 'ls'

    return lagr, loss_type, pde_loss_L2, data_fitting_loss_0, data_fitting_loss_l_r, H_loss



model = PINNModel().to(device)
epochs = 5000

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

dual_opt = ALM(m=1, lr=1e-3, dual_range=(0.,10.), device=device)

lr_schedule = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=100,
    T_mult=1,
    eta_min=0.5*1e-4
)

losses, pde_losses, data_losses_0, bc_losses = [], [], [], []
H_losses_min, H_losses_max, H_losses_mean, H_losses_std = [], [], [], []
H_losses_abs_error, H_losses_rel_error = [], []
t0 = time()

for epoch in range(epochs):
    optimizer.zero_grad()
    loss, loss_type, pde_loss, data_loss_0, bc_loss, H_loss = custom_loss(inputs, model, epoch)
    loss.backward()
    optimizer.step()
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
        
        H0 = H(u_0(x_grid.flatten().reshape(-1, 1)).reshape(Nt, Nx), u_0_x(x_grid.flatten().reshape(-1, 1)).reshape(Nt, Nx), dx)
        Hf = H_loss.detach()
        # breakpoint()
        H_abs_error = torch.abs(Hf - H0)
        H_losses_abs_error.append(torch.max(H_abs_error).item())
        H_rel_error = H_abs_error / (torch.abs(H0) + 1e-16)
        if isinstance(H_rel_error, torch.Tensor):
            H_rel_error = H_rel_error.item() if H_rel_error.numel() == 1 else H_rel_error.max().item()
        H_losses_rel_error.append(H_rel_error)

        if epoch > 1:
            # SoftAdaptive weights update
            # num1 = tf.math.exp(tf.experimental.numpy.cbrt(pde_losses[-1] - pde_losses[-2]))
            # num2 = tf.math.exp(tf.experimental.numpy.cbrt(data_fitting_losses_0[-1] - data_fitting_losses_0[-2]))
            # num3 = tf.math.exp(tf.experimental.numpy.cbrt(data_fitting_losses_l_r[-1] - data_fitting_losses_l_r[-2]))   
            num1 = np.exp((pde_losses[-1] - pde_losses[-2]))
            num2 = np.exp((data_losses_0[-1] - data_losses_0[-2]))
            num3 = np.exp((bc_losses[-1] - bc_losses[-2]))   
            den  = num1 + num2 + num3

            new_lambdas = torch.tensor([num1 / den, num2 / den, num3 / den])
            lambdas = new_lambdas

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6e}")

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
plt.semilogy(bc_losses, label='Periodic Boundary Conditions Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Contributions')
plt.legend()
plt.grid()
plt.savefig('./results/kdv_loss.png', dpi=300) if save_fig else None
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(H_losses_min, label='H_loss_min')
plt.plot(H_losses_max, label='H_loss_max')
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian')
plt.title('Hamiltonian over epochs')
plt.legend()
plt.grid()
plt.savefig('./results/kdv_H_loss.png', dpi=300) if save_fig else None
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
plt.savefig('./results/kdv_H_loss_mean.png', dpi=300) if save_fig else None
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(H_losses_std_arr)
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian std')
plt.title('Hamiltonian standard deviation over epochs')
plt.grid()
plt.savefig('./results/kdv_H_loss_std.png', dpi=300) if save_fig else None
plt.show()

H_losses_abs_error = np.array(H_losses_abs_error)
plt.figure(figsize=(10, 6))
plt.plot(H_losses_abs_error)
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian absolute error')
plt.title('Hamiltonian absolute error over epochs')
plt.grid()
plt.savefig('./results/kdv_H_loss_abs_error.png', dpi=300) if save_fig else None
plt.show()

H_losses_rel_error = np.array(H_losses_rel_error)
plt.figure(figsize=(10, 6))
plt.plot(H_losses_rel_error)
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian relative error')
plt.title('Hamiltonian relative error over epochs')
plt.grid()
plt.savefig('./results/kdv_H_loss_rel_error.png', dpi=300) if save_fig else None
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
ax.set_title('KdV equation')
ax.set_box_aspect(None, zoom=0.85)
plt.savefig('./results/kdv_solution.png', dpi=300) if save_fig else None
plt.show()