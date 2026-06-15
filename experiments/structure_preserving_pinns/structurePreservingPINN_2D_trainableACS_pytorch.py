import numpy as np
import torch
import torch.nn as nn
from time import time
import matplotlib.pyplot as plt

DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Nx, Ny, Nt = 10, 10, 10
N_collocation = Nx*Ny*Nt
d_in = 3

xMin, xMax = 0.0, 8.0
yMin, yMax = 0.0, 8.0
tMax = 50.

def choose_width_depth(N_collocation=N_collocation, overparam_factor=3.0, d_in=d_in, d_out=1):
    N_target = int(overparam_factor * N_collocation)
    depth = 8
    a, b = depth - 1, d_in + depth + d_out
    c = d_out - N_target
    width = int((-b + np.sqrt(b*b - 4*a*c)) / (2*a))
    return width, depth

width, depth = choose_width_depth()
dx = (xMax - xMin) / (Nx - 1)
dy = (yMax - yMin) / (Ny - 1)
dt = tMax / (Nt - 1)
h = np.max([dx, dy, dt])

lambdas = torch.tensor([1., 1., 1.], dtype=DTYPE, device=device, requires_grad=False)
cheb_par = torch.tensor(0.5, dtype=DTYPE, device=device, requires_grad=True)

x = torch.linspace(xMin, xMax, Nx, dtype=DTYPE, device=device).reshape(-1, 1)
y = torch.linspace(yMin, yMax, Ny, dtype=DTYPE, device=device).reshape(-1, 1)
t = torch.linspace(0, tMax, Nt, dtype=DTYPE, device=device).reshape(-1, 1)
y_grid, x_grid, t_grid = torch.meshgrid(y.flatten(), x.flatten(), t.flatten(), indexing='ij')
x_train = x_grid.flatten().reshape(-1, 1)
y_train = y_grid.flatten().reshape(-1, 1)
t_train = t_grid.flatten().reshape(-1, 1)
xyt_train = torch.stack([x_train.flatten(), y_train.flatten(), t_train.flatten()], dim=1)

save_fig = True

def u_0(x, y):
    epsilon = 0.01
    c1, c2 = 0.45, 0.25
    x1, x2 = 2.5, 3.3
    y1 = 0.
    out = 3*c1/(torch.cosh(0.5*torch.sqrt(torch.tensor(c1/epsilon))*((x-x1)**2 + (y-y1)**2)**0.5))**2 
    out += 3*c2/(torch.cosh(0.5*torch.sqrt(torch.tensor(c2/epsilon))*((x-x2)**2 + (y-y1)**2)**0.5))**2
    return out

def periodic_boundary_conditions(model, Nbc=2000):
    x = torch.rand(Nbc, 1, device=device) * (xMax - xMin) + xMin
    y = torch.rand(Nbc, 1, device=device) * (yMax - yMin) + yMin
    t = torch.rand(Nbc, 1, device=device) * tMax
    
    xL = torch.full_like(x, xMin)
    xR = torch.full_like(x, xMax)
    yL = torch.full_like(y, yMin)
    yR = torch.full_like(y, yMax)
    
    uLx = model(torch.cat([xL, y, t], 1))
    uRx = model(torch.cat([xR, y, t], 1))
    uLy = model(torch.cat([x, yL, t], 1))
    uRy = model(torch.cat([x, yR, t], 1))
    
    loss = torch.mean((uLx - uRx)**2 + (uLy - uRy)**2)
    return loss

def H(u, u_x, u_y):
    return torch.sum((u_x**2 + u_y**2)/2 - u**3/6) * dx * dy

def linear_loss_function(tensors, weights):
    stacked = torch.stack(tensors)
    weights = weights / torch.sum(weights)
    loss = torch.sum(weights * stacked)
    return loss, 'ls'

def chebyshev_loss_function(tensors, weights):
    stacked = torch.stack(tensors)
    loss = torch.max(weights * stacked)
    return loss, 'cs'

def augmentedChebyshev_loss_function(tensors, weights):
    par = torch.sigmoid(cheb_par)
    ls = linear_loss_function(tensors, weights)[0]
    cs = chebyshev_loss_function(tensors, weights)[0]
    return par*cs + (1-par)*ls, 'acs'

class FourierFeatures(nn.Module):
    def __init__(self, n_modes=5):
        super().__init__()
        self.n_modes = n_modes
    
    def forward(self, inputs):
        x, y, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
        features = [t]
        for k in range(1, self.n_modes + 1):
            features.append(torch.sin(2*np.pi*k*(x - xMin)/(xMax-xMin)))
            features.append(torch.cos(2*np.pi*k*(x - xMin)/(xMax-xMin)))
            features.append(torch.sin(2*np.pi*k*(y - yMin)/(yMax-yMin)))
            features.append(torch.cos(2*np.pi*k*(y - yMin)/(yMax-yMin)))
        return torch.cat(features, dim=1)

class PINNModel(nn.Module):
    def __init__(self, num_hidden_layers=depth, num_neurons_per_layer=width):
        super().__init__()
        self.ff = FourierFeatures(n_modes=4)
        layers = []
        # input_dim = 3 + 4 * 2 * 4
        input_dim = 17
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim, num_neurons_per_layer))
            layers.append(nn.Tanh())
            input_dim = num_neurons_per_layer
        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.ff(x)
        return self.net(x)

def custom_loss(inputs, model, dual_opt):
    x, y, t = inputs[:, 0:1], inputs[:, 1:2], inputs[:, 2:3]
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    u_model = model(torch.cat([x, y, t], dim=1))
    
    u_t = torch.autograd.grad(u_model.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u_model.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u_model.sum(), y, create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    
    u_xxx = torch.autograd.grad(u_xx.sum(), x, create_graph=True)[0]
    u_xyy = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0]
    
    pde_loss = torch.mean((u_t + u_model * u_x + u_xxx + u_xyy) ** 2)
    
    x_ic = torch.linspace(xMin, xMax, Nx*Ny).reshape(-1, 1).to(device)
    y_ic = torch.linspace(yMin, yMax, Nx*Ny).reshape(-1, 1).to(device)
    t_ic = torch.zeros_like(x_ic)
    u_ic = u_0(x_ic, y_ic)
    u_ic_pred = model(torch.cat([x_ic, y_ic, t_ic], dim=1))
    data_fitting_loss_0 = torch.mean((u_ic_pred - u_ic) ** 2)
    
    data_fitting_loss_l_r = periodic_boundary_conditions(model)
    loss, loss_type = augmentedChebyshev_loss_function([pde_loss, data_fitting_loss_0, data_fitting_loss_l_r], lambdas)
    
    H_loss = H(u_model.reshape(Nx, Ny, Nt), u_x.reshape(Nx, Ny, Nt), u_y.reshape(Nx, Ny, Nt))
    
    # constraint
    H0 = H(u_0(x_grid.flatten().reshape(-1, 1)), u_0_x(x_grid.flatten().reshape(-1, 1)), dx)

    Hf = H(u_model.reshape(Nt, Nx), u_x.reshape(Nt, Nx), dx)
    H_constraint = torch.abs(Hf - H0)/torch.abs(H0)

    eps = 5/(epoch+1)
    H_constraint = torch.max(H_constraint - eps, torch.zeros_like(H_constraint)).unsqueeze(0)

    loss = dual_opt.forward_update(loss, H_constraint)
    
    
    return loss, loss_type, pde_loss, data_fitting_loss_0, data_fitting_loss_l_r, H_loss

model = PINNModel().to(device)
epochs = 1000
lr_schedule = torch.optim.lr_scheduler.PolynomialLR(
    torch.optim.Adam(model.parameters(), lr=1e-2),
    total_iters=epochs, power=3.0
)
optimizer = lr_schedule.optimizer

losses, pde_losses, data_losses_0, bc_losses = [], [], [], []
H_losses_min, H_losses_max, H_losses_mean, H_losses_std = [], [], [], []
H_losses_abs_error, H_losses_rel_error = [], []
t0 = time()

for epoch in range(epochs):
    optimizer.zero_grad()
    loss, loss_type, pde_loss, data_loss_0, bc_loss, H_loss = custom_loss(xyt_train, model)
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
        
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.6e}")

print(f'\nComputation time: {time() - t0:.2f}s')

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
plt.savefig('./results/2D_loss.png', dpi=300) if save_fig else None
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(H_losses_min, label='H_loss_min')
plt.plot(H_losses_max, label='H_loss_max')
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian')
plt.title('Hamiltonian over epochs')
plt.legend()
plt.grid()
plt.savefig('./results/2D_H_minmax.png', dpi=300) if save_fig else None
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
plt.savefig('./results/2D_H_mean_std.png', dpi=300) if save_fig else None
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(H_losses_std_arr)
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian std')
plt.title('Hamiltonian standard deviation over epochs')
plt.grid()
plt.savefig('./results/2D_H_std.png', dpi=300) if save_fig else None
plt.show()

if cheb_par.requires_grad:
    plt.figure(figsize=(10, 6))
    plt.plot(torch.sigmoid(cheb_par).detach())
    plt.xlabel('Epoch')
    plt.ylabel('Chebyshev parameter')
    plt.title('Chebyshev parameter over epochs')
    plt.grid()
    plt.savefig('./results/2D_cheb_par.png', dpi=300) if save_fig else None
    plt.show()

from mpl_toolkits.mplot3d import Axes3D
N = 100
xspace = torch.linspace(xMin, xMax, N, dtype=DTYPE, device=device)
yspace = torch.linspace(yMin, yMax, N, dtype=DTYPE, device=device)
tspace_val = torch.tensor(tMax, dtype=DTYPE, device=device)
X_grid, Y_grid = torch.meshgrid(xspace, yspace, indexing='ij')
T_grid = torch.full_like(X_grid, tMax)
XYTgrid = torch.stack([X_grid.flatten(), Y_grid.flatten(), T_grid.flatten()], dim=1)

with torch.no_grad():
    u_pred = model(XYTgrid)
U = u_pred.reshape(N, N)

X_np = X_grid.cpu().numpy()
Y_np = Y_grid.cpu().numpy()
U_np = U.cpu().numpy()

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_np, Y_np, U_np, cmap='viridis')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$u(x,y,t)$')
ax.set_title('2D PDE Solution')
plt.savefig('./results/2D_solution.png', dpi=300) if save_fig else None
plt.show()
