import numpy as np
import tensorflow as tf
from time import time
import matplotlib.pyplot as plt

# TF_USE_LEGACY_KERAS=True

DTYPE = np.float32
# Nx = 50
# Nt = 50
# N_collocation = Nx*Nt

xMin = -np.pi
xMax = np.pi
tMax = 5. # 10.
d_in = 2

def Nx_from_arch(width, depth, fac=2.0, d_in=2, d_out=1):
    """
    Given a PINN architecture (width, depth) and an overparam factor fac,
    compute Nx = Nt such that:
    
        Nx * Nt  ≈ N_params / fac,
        Nx = Nt,
        
    where N_params is the number of trainable parameters.
    
    Parameters
    ----------
    width : int
        Number of neurons per hidden layer.
    depth : int
        Number of hidden layers.
    fac : float
        Over-parameterization factor. Typical values: fac = 2 or 3.
    d_in : int
        Input dimension (usually 2: x,t).
    d_out : int
        Output dimension (usually 1: u).
        
    Returns
    -------
    Nx : int
    Nt : int
    Ntheta : int
        Total number of trainable parameters.
    Ncoll_target : int
        Target number of collocation points = Ntheta/fac.
    """

    # Parameter count
    Ntheta = (d_in + 1) * width \
             + (depth - 1) * (width * width + width) \
             + d_out * (width + 1)

    # Target collocation count
    Ncoll_target = int(Ntheta / fac)

    # Square grid Nx = Nt
    Nx = int(np.sqrt(Ncoll_target))
    Nt = Nx

    return Nx, Nt, Ntheta, Ncoll_target

width = 80
depth = 4

Nx, Nt, Ntheta, Ncoll = Nx_from_arch(width=width, depth=depth, fac=10.)

def h_from_NxNt(Nx, Nt, xMin, xMax, tMax):
    """
    Compute dx, dt, and h from Nx, Nt and the domain extents.
    h is defined as max(dx, dt).

    Returns
    -------
    dx : float
    dt : float
    h  : float
    """

    Lx = xMax - xMin
    Lt = tMax

    dx = Lx / (Nx - 1)
    dt = Lt / (Nt - 1)

    h = max(dx, dt)

    return dx, dt, h

dx, dt, h = h_from_NxNt(Nx, Nt, xMin, xMax, tMax)

lambdas = [1., 1., 1.]
lambdas = tf.Variable(lambdas, trainable=False, name='lambdas', dtype=DTYPE)
do_training = True
cheb_par = tf.Variable(0.5, trainable=False, name='cheb_par', dtype=DTYPE)

x = np.linspace(xMin, xMax, Nx).reshape((-1, 1)).astype(DTYPE)
t = np.linspace(0, tMax, Nt).reshape((-1, 1)).astype(DTYPE)

x_train = tf.expand_dims(tf.convert_to_tensor(x.flatten()), axis=-1)
t_train = tf.expand_dims(tf.convert_to_tensor(t.flatten()), axis=-1)

save_fig = True

# Define the initial condition
def u_0(x):
    return 0.2+0.1*tf.math.cos(2 * x)


def u_0_x(x):
    return -0.2*tf.math.sin(2 * x)


def periodic_boundary_conditions(model, Nbc=2000):

    # Random boundary sampling (correct choice)
    x = tf.random.uniform((Nbc,1), xMin, xMax)
    t = tf.random.uniform((Nbc,1), 0.0, tMax)

    xL = tf.ones_like(x) * xMin
    xR = tf.ones_like(x) * xMax

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([xL, xR])

        uLx = model(tf.concat([xL, t], 1))
        uRx = model(tf.concat([xR, t], 1))

    # First derivatives
    uxL = tape.gradient(uLx, xL)
    uxR = tape.gradient(uRx, xR)

    del tape

    # Enforce periodicity of values AND derivatives
    loss = tf.reduce_mean(
        (uLx - uRx)**2 +
        (uxL - uxR)**2 
    )

    return loss


# def H(u, u_x, dx):
#     return tf.reduce_sum(tf.pow(u, 3)+u*tf.pow(u_x, 2), axis = -1) * dx
#     # return tf.reduce_sum((tf.pow(u, 2)+tf.pow(u_x, 2))/2, axis = -1) * dx
    
def ch_density(u, u_x):
    return tf.pow(u, 3) + u * tf.pow(u_x, 2)


# @tf.function
def H(u, u_x, dx, density_fn=ch_density, axis=-1):
    """
    Boole’s rule (8th order) along 'axis' for uniform grid with spacing dx.
    Requires (N-1) % 4 == 0. Otherwise uses Boole on the largest prefix and trapezoid on remainder.
    """
    f = density_fn(u, u_x)   # [..., N]
    n = tf.shape(f)[axis]

    # Trapezoid as a fallback on short tails
    def _trap_rem(rem):
        # rem: [..., M] contiguous tail; integrate with trapezoid
        return tf.reduce_sum(0.5*(rem[..., 1:] + rem[..., :-1]), axis=-1) * tf.cast(dx, f.dtype)

    # Degenerate
    if tf.less_equal(n, 1):
        return tf.reduce_sum(f, axis=axis) * dx

    # Largest prefix with (n1-1) % 4 == 0
    n1 = n - ((n - 1) % 4)
    # Boole constant for uniform spacing: 2*dx/45
    c = (2.0 * dx) / 45.0

    # Indices for prefix
    idx_prefix = tf.range(n1)
    f0   = tf.gather(f, idx_prefix[0::4], axis=axis)      # 0,4,8,...
    f1   = tf.gather(f, idx_prefix[1::4], axis=axis)      # 1,5,9,...
    f2   = tf.gather(f, idx_prefix[2::4], axis=axis)      # 2,6,10,...
    f3   = tf.gather(f, idx_prefix[3::4], axis=axis)      # 3,7,11,...
    f4   = tf.gather(f, idx_prefix[4::4], axis=axis)      # 4,8,12,... (last block end)

    # Weighted sum across blocks
    # Boole's block weights per 5 nodes: [7, 32, 12, 32, 7]
    # Aggregate across all blocks by summing slices
    s = 7.0  * tf.reduce_sum(f0, axis=axis)
    s += 32.0 * tf.reduce_sum(f1, axis=axis)
    s += 12.0 * tf.reduce_sum(f2, axis=axis)
    s += 32.0 * tf.reduce_sum(f3, axis=axis)
    s += 7.0  * tf.reduce_sum(f4, axis=axis)

    boole_part = c * s

    # Tail remainder
    if tf.equal(n1, n):
        return boole_part

    rem = tf.gather(f, tf.range(n1-1, n), axis=axis)  # nodes: n1-1 .. n-1
    tail = _trap_rem(rem)
    return boole_part + tail    


def linear_loss_function(tensors, weights):
    """
    Computes the sum of the input tensors.
    
    Args:
        tensors (list of tf.Tensor): List of tensors to compute the sum.
        
    Returns:
        tf.Tensor: The sum of the input tensors.
    """
    # weights = weights / tf.reduce_sum(weights)  # Normalize weights
    # stack_tensor = tf.stack(tf.multiply(tensors, weights))
    # stack_tensor = tf.stack([w * t for w, t in zip(weights, tensors)], axis=0)   
    stacked = tf.stack(tensors, axis=0)   # shape (n_losses,)
    weights = weights / tf.reduce_sum(weights)
    loss = tf.reduce_sum(weights * stacked)  
    loss_type = 'ls'
    return loss, loss_type


def chebyshev_loss_function(tensors, weights):
    """
    Computes the max of the input tensors.
    
    Args:
        tensors (list of tf.Tensor): List of tensors to compute the log-sum-exp.
        
    Returns:
        tf.Tensor: The maximum of the input tensors.
    """
    # weights = weights / tf.reduce_sum(weights)  # Normalize weights
    # stack_tensor = tf.stack(tf.multiply(tensors, weights))
    # stack_tensor = tf.stack([w * t for w, t in zip(weights, tensors)], axis=0)
    stacked = tf.stack(tensors, axis=0)
    loss = tf.reduce_max(weights*stacked)
    loss_type = 'cs'
    return loss, loss_type


def smooth_chebyshev_loss_function(mu, tensors, weights):
    """
    Computes the log of the sum of the exponentials of the input tensors.
    
    Args:
        tensors (list of tf.Tensor): List of tensors to compute the log-sum-exp.
        
    Returns:
        tf.Tensor: The log-sum-exp of the input tensors.
    """
    weights = weights / tf.reduce_sum(weights)  # Normalize weights
    # stack_tensor = tf.stack(tf.multiply(tensors, weights))
    # stack_tensor = tf.stack([w * t for w, t in zip(weights, tensors)], axis=0)
    stacked = tf.stack(tensors, axis=0)
    exp_sum = tf.reduce_sum(tf.math.exp(stacked/mu), axis=0)
    loss = mu*tf.math.log(exp_sum)
    loss_type = 'scs'
    return loss, loss_type


def augmentedChebyshev_loss_function(tensors, weights):
    """
    Computes the log of the sum of the exponentials of the input tensors.
    
    Args:
        tensors (list of tf.Tensor): List of tensors to compute the log-sum-exp.
        
    Returns:
        tf.Tensor: The log-sum-exp of the input tensors.
    """
    # weights = weights / tf.reduce_sum(weights)  # Normalize weights
    loss_type = 'acs'
    par = tf.sigmoid(cheb_par)  # par is between 0 and 1
    return par*chebyshev_loss_function(tensors, weights)[0] + (1-par)*linear_loss_function(tensors, weights)[0], loss_type


def sigmoid_centered(x):
    return 2*tf.nn.sigmoid(.5*x) - 1

def PINNModel(num_hidden_layers=depth, num_neurons_per_layer=width):  # 8,40 
    xt_input = tf.keras.Input(shape=(2,))
    output_u = xt_input
    for _ in range(num_hidden_layers):
        output_u = tf.keras.layers.Dense(num_neurons_per_layer,
                                         activation='gelu',  # tanh
                                         kernel_initializer='glorot_normal', #'glorot_uniform',  # glorot_normal
                                         )(output_u)

    output_u = tf.keras.layers.Dense(units=1,
                                     activation='linear',
                                     kernel_initializer='glorot_normal', #'glorot_uniform',  # glorot_normal
                                     )(output_u)

    return tf.keras.Model(inputs=xt_input, outputs=output_u)  #tf.keras.Model(inputs=[x_input, t_input], outputs=output_u)


def lambda_grad(epoch,
                start=1000,
                lam_max=1e-0,
                kappa=1e-3):
    epoch = tf.cast(epoch, tf.float32)
    return lam_max * (1.0 - tf.exp(-kappa * tf.maximum(epoch - start, 0.0)))



# @tf.function
def custom_loss(inputs, model):
    x, t = inputs[:, 0:1], inputs[:, 1:2]

    with tf.GradientTape(persistent=True) as outerTape:
        outerTape.watch(x)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(x)
            with tf.GradientTape(persistent=False) as tape2:            
                tape2.watch(x)
                tape2.watch(t)
                with tf.GradientTape(persistent=True) as tape3:
                    tape3.watch(x)
                    tape3.watch(t)
                    u_model = model(tf.stack([x[:, 0], t[:, 0]], axis=1))
                u_x = tape3.gradient(u_model, x)
                u_t = tape3.gradient(u_model, t)
            u_xx = tape2.gradient(u_x, x)     
        u_xxt = tape.gradient(u_xx, t)   
        u_xxx = tape.gradient(u_xx, x) 

        # === Camassa–Holm residual ===
        r = (
            u_t
            - u_xxt
            + 3.0 * u_model * u_x
            - 2.0 * u_x * u_xx
            - u_model * u_xxx
        )
    r_x = outerTape.gradient(r, x)

    # Clean up
    del tape, tape2, tape3, outerTape
    
    lam = lambda_grad(epoch)

    # === H1 norm of residual ===
    pde_loss_L2 = tf.reduce_mean(tf.square(r))
    pde_loss_grad = tf.reduce_mean(tf.square(r_x))
    pde_loss_H1 = pde_loss_L2 + lam * pde_loss_grad

    # === Initial condition ===
    ic_mask = tf.where(tf.abs(t) < 1e-6)
    x_ic = tf.gather(x, ic_mask[:, 0])
    u_ic = u_0(x_ic)
    t_ic = tf.zeros_like(x_ic)
    u_ic_pred = model(tf.concat([x_ic, t_ic], axis=1))
    data_fitting_loss_0 = tf.reduce_mean(tf.square(u_ic_pred - u_ic))

    # === Periodic boundary conditions ===
    data_fitting_loss_l_r = periodic_boundary_conditions(model)

    # === Chebyshev aggregation ===
    loss, loss_type = chebyshev_loss_function(
        [pde_loss_H1, data_fitting_loss_0, data_fitting_loss_l_r],
        lambdas
    )

    # === Hamiltonian (monitor only) ===
    H_loss = H(
        tf.reshape(u_model, shape=[Nt, Nx]),
        tf.reshape(u_x, shape=[Nt, Nx]),
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
    

# Create the PINN model
model = PINNModel()
model.summary()

epochs = 1000  # 3000  # 5000  # 2000
# # Compile the model
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#               loss=lambda y_true, y_pred: custom_loss([x_train, t_train, theta_train], model)[1])

# Create the optimizer with a smaller learning rate
# learning_rate = 1e-3  # 1e-4
# learning_rate_type = 'constant'
# learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([10, 100], [1e-1, 5e-2, 1e-2])  #OK
# learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([100, 300], [1e-2, 1e-3, 1e-4])
# learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
#     initial_learning_rate=1e-3,
#     decay_steps=epochs,
#     end_learning_rate=1e-5,
#     power=2.,
#     cycle=False,  # True    
#     name='PolynomialDecay'
# )
# learning_rate_type = 'polynomialDecay'
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=epochs, # 100
    decay_rate=0.9,
    staircase=False,
    name='ExponentialDecay'
)
learning_rate_type = 'exponentialDecay'
# learning_rate = tf.keras.optimizers.schedules.CosineDecay(
#     initial_learning_rate=1e-3,
#     decay_steps=1000,
#     alpha=0.0,
#     name='CosineDecay',
#     warmup_target=None,
#     warmup_steps=0
# )
# learning_rate_type = 'cosineDecay'

trainable = model.trainable_variables 
if lambdas.trainable:
    trainable += [lambdas]
    
if cheb_par.trainable:
    trainable += [cheb_par]

# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-08, amsgrad=True)
# optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate, beta_1=0.8, beta_2=0.9, epsilon=1e-07)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False)

# Training loop
losses = []
pde_losses = []
data_fitting_losses_0 = []
data_fitting_losses_l_r = []
delta_gradients = []
H_losses_min = []
H_losses_max = []
H_losses_mean = []
H_losses_std = []
H_losses_abs_error = []
H_losses_rel_error = []
lambdas_values = []
lambdas_values.append(lambdas.numpy())
cheb_par_values = []
cheb_par_values.append(cheb_par.numpy())

# Convert data to tensor because tf.GradientTape() can only watch tensor and not numpy arrays
x_train = tf.convert_to_tensor(x_train)
t_train = tf.convert_to_tensor(t_train)
x_grid, t_grid = np.meshgrid(x.flatten(), t.flatten())
inputs = tf.convert_to_tensor(np.vstack([x_grid.flatten(), t_grid.flatten()]).T)
stop = False
# Start timer
t0 = time()
for epoch in range(epochs):
    if not stop:
        # print("# STARTING EPOCH", epoch + 1)

        # Create a LearningRateScheduler to update the learning rate
        # current_lr = scheduler(epoch, learning_rate)
        # tf.keras.backend.set_value(optimizer.lr, current_lr)

        with tf.GradientTape() as tape:
            loss, loss_type, pde_loss, data_fitting_loss_0, data_fitting_loss_l_r, H_loss = custom_loss(inputs, model)

        # print("Computing gradients")
        gradients = tape.gradient(loss, trainable)
        # print(gradients[-1])
        # print("Applying gradients")
        optimizer.apply_gradients(zip(gradients, trainable))
        # print("Appending losses")
        losses.append(loss.numpy())
        pde_losses.append(pde_loss.numpy())
        data_fitting_losses_0.append(data_fitting_loss_0.numpy())
        data_fitting_losses_l_r.append(data_fitting_loss_l_r.numpy())
        H_loss_min = tf.reduce_min(H_loss)
        H_loss_max = tf.reduce_max(H_loss)
        H_losses_min.append(H_loss_min.numpy())
        H_losses_max.append(H_loss_max.numpy())
        H_loss_mean = tf.reduce_mean(H_loss)
        H_loss_std = tf.math.reduce_std(H_loss)
        H_losses_mean.append(H_loss_mean.numpy())
        H_losses_std.append(H_loss_std.numpy())
        
        H0 = H(u_0(x_grid), u_0_x(x_grid), dx)  # H0 = H_loss[0].numpy()
        Hf = H_loss.numpy()
        H_abs_error = tf.abs(Hf - H0)
        H_losses_abs_error.append(tf.reduce_max(H_abs_error).numpy())
        H_rel_error = H_abs_error / tf.abs((H0 + 1e-16))
        H_losses_rel_error.append(H_rel_error[-1].numpy())
        
        # lambdas_values.append((trainable[-1]).numpy())
        if len(losses) > 1 and not lambdas.trainable and do_training:
            # SoftAdaptive weights update
            num1 = tf.math.exp(pde_losses[-1] - pde_losses[-2])
            num2 = tf.math.exp(data_fitting_losses_0[-1] - data_fitting_losses_0[-2])
            num3 = tf.math.exp(data_fitting_losses_l_r[-1] - data_fitting_losses_l_r[-2])
            den  = num1 + num2 + num3

            new_lambdas = tf.stack([num1 / den, num2 / den, num3 / den])
            lambdas.assign(new_lambdas)
            lambdas_values.append((lambdas).numpy())
            
        if cheb_par.trainable:
            cheb_par_values.append(cheb_par.numpy())

        del tape

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")

        if len(losses) > 2 and np.abs(losses[-1] - losses[-2]) / np.abs(losses[-2]) < 1e-8:
            stop = True

print(f"Loss type: {loss_type}")
print(f"Hamiltonian mean: {H_loss_mean.numpy()}")
print(f"Hamiltonian standard deviation: {H_loss_std.numpy()}")
print(f"Hamiltonian maximum: {H_loss_max.numpy()}")
print(f"Hamiltonian minimum: {H_loss_min.numpy()}")
# print(f"Hamiltonian absolute error: {H_abs_error.numpy()}")
# print(f"Hamiltonian relative error: {H_rel_error.numpy()}")
print(f"Hamitonian relative error: {H_rel_error[-1].numpy()}")
# Print computation time
print('\nComputation time: {} seconds'.format(time() - t0))


def generate_save_fig_string(type, epochs, learning_rate_type, loss_type):
    """
    Generates a string for saving figures that includes the number of epochs and the type of learning rate.
    
    Args:
        epochs (int): The number of epochs.
        learning_rate_type (str): The type of learning rate.
        
    Returns:
        str: The generated string for saving figures.
    """
    return f"./results/{type}_epochs_{epochs}_lr_{learning_rate_type}_{loss_type}.png"

# Plot the loss history
plt.semilogy(losses, label='Total Loss')
plt.semilogy(pde_losses, label='PDE Loss')
plt.semilogy(data_fitting_losses_0, label='Initial Conditions Loss')
plt.semilogy(data_fitting_losses_l_r, label='Boundary Conditions Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Contributions')
plt.legend()
plt.grid()

if save_fig:
    save_fig_string = generate_save_fig_string('loss', epochs, learning_rate_type, loss_type)
    # save png
    plt.savefig(save_fig_string, dpi=300)
plt.show()
    # # save pdf
    # plt.savefig('../results/' + 'loss.pdf', dpi=300)

# Evaluate the function
x_eval = np.linspace(x_train[0].numpy(), x_train[-1].numpy(), 100).reshape((-1, 1)).astype(np.float32)
t_eval = np.linspace(t_train[0].numpy(), t_train[-1].numpy(), 100).reshape((-1, 1)).astype(np.float32)
inputs_eval = [x_eval, t_eval]

# Plot the Hamiltonian over epochs
plt.plot(H_losses_min, label='min H')
plt.plot(H_losses_max, label='max H')
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian')
plt.title('Hamiltonian over epochs')
plt.legend()
plt.grid()

if save_fig:
    save_fig_string = generate_save_fig_string('H_loss', epochs, learning_rate_type, loss_type)
    # save png
    plt.savefig(save_fig_string, dpi=300)
plt.show()
    # # save pdf
    # plt.savefig('../results/' + 'H_loss.pdf', dpi=300)    
    
# Plot the average Hamiltonian over epochs with standard deviation
H_losses_mean = np.array(H_losses_mean)
H_losses_std = np.array(H_losses_std)
H_losses_rel_error = np.array(H_losses_rel_error)
H_losses_rel_error = np.array(H_losses_rel_error)

plt.plot(H_losses_mean)
plt.fill_between(range(len(H_losses_mean)), H_losses_mean - H_losses_std, H_losses_mean + H_losses_std, alpha=0.2)
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian mean')
plt.title('Hamiltonian mean over epochs with standard deviation')
# plt.legend()
plt.grid()

if save_fig:
    save_fig_string = generate_save_fig_string('H_loss_mean', epochs, learning_rate_type, loss_type)
    # save png
    plt.savefig(save_fig_string, dpi=300)
plt.show()
    # # save pdf
    # plt.savefig('../results/' + 'H_loss_std.pdf', dpi=300)

# Plot the standard deviation of the Hamiltonian over epochs
plt.plot(H_losses_std)
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian std')
plt.title('Hamiltonian standard deviation over epochs')
# plt.legend()
plt.grid()

if save_fig:
    save_fig_string = generate_save_fig_string('H_loss_std', epochs, learning_rate_type, loss_type)
    # save png
    plt.savefig(save_fig_string, dpi=300)
plt.show()
    # # save pdf
    # plt.savefig('../results/' + 'H_loss_std.pdf', dpi=300)
    
# Plot the absolute error of the Hamiltonian over epochs
plt.plot(H_losses_abs_error)    
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian absolute error')
plt.title('Hamiltonian absolute error over epochs')
# plt.legend()
plt.grid()

if save_fig:
    save_fig_string = generate_save_fig_string('H_loss_abs_error', epochs, learning_rate_type, loss_type)
    # save png
    plt.savefig(save_fig_string, dpi=300)
plt.show()
    # # save pdf
    # plt.savefig('../results/' + 'H_loss_rel_error.pdf', dpi=300)
    

# Plot the relative error of the Hamiltonian over epochs
plt.plot(H_losses_rel_error)    
plt.xlabel('Epoch')
plt.ylabel('Hamiltonian relative error')
plt.title('Hamiltonian relative error over epochs')
# plt.legend()
plt.grid()

if save_fig:
    save_fig_string = generate_save_fig_string('H_loss_rel_error', epochs, learning_rate_type, loss_type)
    # save png
    plt.savefig(save_fig_string, dpi=300)
plt.show()
    # # save pdf
    # plt.savefig('../results/' + 'H_loss_rel_error.pdf', dpi=300)


# Plot the Chebyshev parameter over epochs
if cheb_par.trainable:
    plt.plot(tf.sigmoid(cheb_par_values))
    plt.xlabel('Epoch')
    plt.ylabel('Chebyshev parameter')
    plt.title('Chebyshev parameter over epochs')
    plt.grid()
    if save_fig:
        save_fig_string = generate_save_fig_string('cheb_par', epochs, learning_rate_type, loss_type)
        # save png
        plt.savefig(save_fig_string, dpi=300)
plt.show()
        # # save pdf
        # plt.savefig('../results/' + 'cheb_par.pdf', dpi=300)    
    
    
from mpl_toolkits.mplot3d import Axes3D

# Set up meshgrid
N = 600
tspace = np.linspace(0, tMax, N + 1)
xspace = np.linspace(xMin, xMax, N + 1)
T, X = np.meshgrid(tspace, xspace)
XTgrid = np.vstack([X.flatten(),T.flatten()]).T

# Determine predictions of u(t, x)
u_pred = model(tf.cast(XTgrid,DTYPE))

# Reshape upred
U = u_pred.numpy().reshape(N+1,N+1)

# Surface plot of solution u(t,x)
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, U, cmap='viridis')
ax.view_init(35,35)
ax.set_xlabel('$x$')
ax.set_ylabel('$t$')
ax.set_zlabel('$u_\\theta(x,t)$')
ax.set_title('Solution to Camassa-Holm equation')
ax.set_box_aspect(None, zoom=0.85)

if save_fig:
    save_fig_string = generate_save_fig_string('sol', epochs, learning_rate_type, loss_type)
    # save png
    plt.savefig(save_fig_string, dpi=300)
plt.show()
    # # save pdf
    # plt.savefig('../results/' + 'solution.pdf', dpi=300)




import pandas as pd
df = pd.DataFrame()

df['total_loss'] = losses
df['pde_loss'] = pde_losses
df['data_fitting_loss_0'] = data_fitting_losses_0
df['data_fitting_loss_l_r'] = data_fitting_losses_l_r
df['H_loss_min'] = H_losses_min
df['H_loss_max'] = H_losses_max
df['H_loss_mean'] = H_losses_mean
df['H_loss_std'] = H_losses_std
df['H_loss_abs_error'] = H_losses_abs_error
df['H_loss_rel_error'] = H_losses_rel_error
# df['cheb_par'] = cheb_par_values

df.to_csv('./results/camassa/training_history.csv', index=False)