import numpy as np
from sympy import N
import tensorflow as tf
from time import time
import matplotlib.pyplot as plt

# TF_USE_LEGACY_KERAS=True

DTYPE = np.float32
Nx = 10 #100
Ny = 10 #100
Nt = 10 #50
N_collocation = Nx*Ny*Nt
d_in = 3

xMin = 0.0
xMax = 8.0
yMin = 0.0
yMax = 8.0
tMax = 50.
d_in = 2

def choose_width_depth(N_collocation=N_collocation, overparam_factor=3.0, d_in=d_in, d_out=1):
    """
    Returns a single (width, depth) pair for a mildly overparameterized PINN.
    
    N_collocation : Nx * Ny * Nt
    overparam_factor : how many times larger the model than data (default 3x)
    """
    
    N_target = int(overparam_factor * N_collocation)

    # For 3rd-order PDEs like ZK:
    # depth 5–7 is stable. We fix depth=6 (good compromise).
    depth = 8

    # Parameter formula:
    # N = (depth-1) w^2 + (d_in + depth + d_out) w + d_out
    a = depth - 1
    b = d_in + depth + d_out
    c = d_out - N_target

    # Solve quadratic for width
    width = int((-b + np.sqrt(b*b - 4*a*c)) / (2*a))

    return width, depth

width, depth = choose_width_depth()
dx = (xMax - xMin) / (Nx - 1)
dy = (yMax - yMin) / (Ny - 1)
dt = tMax / (Nt - 1)
h = np.max([dx, dy, dt])

lambdas = [1., 1., 1.] #[1.7, 0.2, 1.4] #for cs  #[1., 0.5, 1.]  # [1.7, 0.2, 1.4] for cs  #
lambdas = tf.Variable(lambdas, trainable=False, name='lambdas', dtype=DTYPE)

cheb_par = tf.Variable(0.5, trainable=True, name='cheb_par', dtype=DTYPE)

x = np.linspace(xMin, xMax, Nx).reshape((-1, 1)).astype(DTYPE)
y = np.linspace(yMin, yMax, Ny).reshape((-1, 1)).astype(DTYPE)
t = np.linspace(0, tMax, Nt).reshape((-1, 1)).astype(DTYPE)
x_grid, y_grid, t_grid = np.meshgrid(x, y, t, indexing='ij')
x_train = x_grid.flatten(); x_train = tf.convert_to_tensor(x_train); x_train = tf.expand_dims(x_train, axis=-1)
y_train = y_grid.flatten(); y_train = tf.convert_to_tensor(y_train); y_train = tf.expand_dims(y_train, axis=-1)
t_train = t_grid.flatten(); t_train = tf.convert_to_tensor(t_train); t_train = tf.expand_dims(t_train, axis=-1)
xyt_train = tf.concat([x_train, y_train, t_train], axis=-1)

save_fig = True

# Define the initial condition
def u_0(x, y):
    ##1
    epsilon = 0.01
    theta = 0.
    y1 = 0.
    y2 = 0.
    c1 = 0.45
    c2 = 0.25
    x1 = 2.5
    x2 = 3.3
    out = 3*c1/(tf.math.cosh(0.5*tf.sqrt(c1/epsilon)*((x-x1)*tf.math.cos(theta) + (y-y1)*tf.math.sin(theta))))**2 
    + 3*c2/(tf.math.cosh(0.5*tf.sqrt(c2/epsilon)*((x-x2)*tf.math.cos(theta) + (y-y2)*tf.math.sin(theta))))**2
    ##2
    # epsilon = 0.01
    # theta = 0.
    # y1 = 4.
    # c1 = 1.
    # x1 = 2.5
    # out = 3*c1/(tf.math.cosh(0.5*tf.sqrt(c1/epsilon)*((x-x1)*tf.math.cos(theta) + (y-y1)*tf.math.sin(theta))))**2

    return out
 # mpmath for sech

# def periodic_boundary_conditions(model, Nbc=2000):
#     x = tf.random.uniform((Nbc,1), xMin, xMax)
#     y = tf.random.uniform((Nbc,1), yMin, yMax)
#     t = tf.random.uniform((Nbc,1), 0, tMax)

#     xL = tf.ones_like(x)*xMin; xR = tf.ones_like(x)*xMax
#     yL = tf.ones_like(y)*yMin; yR = tf.ones_like(y)*yMax

#     uLx = model(tf.concat([xL,y,t],1))
#     uRx = model(tf.concat([xR,y,t],1))
#     uLy = model(tf.concat([x,yL,t],1))
#     uRy = model(tf.concat([x,yR,t],1))

#     return tf.reduce_mean((uLx-uRx)**2 + (uLy-uRy)**2)


def periodic_boundary_conditions(model, Nbc=2000):

    # Random boundary sampling (correct choice)
    x = tf.random.uniform((Nbc,1), xMin, xMax)
    y = tf.random.uniform((Nbc,1), yMin, yMax)
    t = tf.random.uniform((Nbc,1), 0.0, tMax)

    xL = tf.ones_like(x) * xMin
    xR = tf.ones_like(x) * xMax
    yL = tf.ones_like(y) * yMin
    yR = tf.ones_like(y) * yMax

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([xL, xR, yL, yR])

        uLx = model(tf.concat([xL, y, t], 1))
        uRx = model(tf.concat([xR, y, t], 1))

        uLy = model(tf.concat([x, yL, t], 1))
        uRy = model(tf.concat([x, yR, t], 1))

    # First derivatives
    uxL = tape.gradient(uLx, xL)
    uxR = tape.gradient(uRx, xR)

    uyL = tape.gradient(uLy, yL)
    uyR = tape.gradient(uRy, yR)

    del tape

    # Enforce periodicity of values AND derivatives
    loss = tf.reduce_mean(
        (uLx - uRx)**2 +
        (uLy - uRy)**2 +
        (uxL - uxR)**2 +
        (uyL - uyR)**2
    )

    return loss



def H(u, u_x, u_y):
    return tf.reduce_sum((tf.pow(u_x,2) + tf.pow(u_y,2))/2.0-tf.pow(u,3)/6.0, axis=[0,1]) * dx*dy


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
        weights (list of tf.Tensor): List of weights for each tensor.
    Returns:
        tf.Tensor: The log-sum-exp of the input tensors.
    """
    # weights = weights / tf.reduce_sum(weights)  # Normalize weights
    loss_type = 'acs'
    par = tf.sigmoid(cheb_par)  # par is between 0 and 1
    return par*chebyshev_loss_function(tensors, weights)[0] + (1-par)*linear_loss_function(tensors, weights)[0], loss_type


class FourierFeatures(tf.keras.layers.Layer):
    def __init__(self, n_modes=5):
        super().__init__()
        self.n_modes = n_modes

    def call(self, inputs):
        x = inputs[:, 0:1]
        y = inputs[:, 1:2]
        t = inputs[:, 2:3]

        features = [t]

        for k in range(1, self.n_modes + 1):
            features.append(tf.sin(2*np.pi*k*(x - xMin)/(xMax-xMin)))
            features.append(tf.cos(2*np.pi*k*(x - xMin)/(xMax-xMin)))
            features.append(tf.sin(2*np.pi*k*(y - yMin)/(yMax-yMin)))
            features.append(tf.cos(2*np.pi*k*(y - yMin)/(yMax-yMin)))

        return tf.concat(features, axis=1)
    
def PINNModel(num_hidden_layers=depth, num_neurons_per_layer=width):  # 8,80 OK (# 8,40  # 10,40)  
    xyt_input = tf.keras.Input(shape=(3,))
    output_u = FourierFeatures(n_modes=4)(xyt_input)
    for _ in range(num_hidden_layers):
        output_u = tf.keras.layers.Dense(num_neurons_per_layer,
                                         activation='tanh',  # tanh
                                         kernel_initializer='glorot_uniform',  # glorot_normal
                                         )(output_u)

    output_u = tf.keras.layers.Dense(units=1,
                                     activation='linear',  # mish
                                     kernel_initializer='glorot_uniform',  # glorot_normal
                                     )(output_u)
    
    return tf.keras.Model(inputs=xyt_input, outputs=output_u)  #tf.keras.Model(inputs=[x_input, t_input], outputs=output_u)


# def PINNModel(num_hidden_layers=depth, num_neurons_per_layer=width):  # 8,80 OK (# 8,40  # 10,40)
#     xyt_input = tf.keras.Input(shape=(3,))
#     output_u = xyt_input
#     for _ in range(num_hidden_layers):
#         output_u = tf.keras.layers.Dense(num_neurons_per_layer,
#                                          activation='tanh',  # tanh
#                                          kernel_initializer='glorot_uniform',  # glorot_normal
#                                          )(output_u)

#     output_u = tf.keras.layers.Dense(units=1,
#                                      activation='linear',  # mish
#                                      kernel_initializer='glorot_uniform',  # glorot_normal
#                                      )(output_u)
    
#     # Define the initial condition
#     # x_input = tf.reshape(xt_input[:, 0], shape=[-1, 1])
#     # t_input = tf.reshape(xt_input[:, 1], shape=[-1, 1])
#     # initial_u = u_0(x_input)
#     # output_u = tf.where(tf.equal(t_input, 0), initial_u, output_u)

#     return tf.keras.Model(inputs=xyt_input, outputs=output_u)  #tf.keras.Model(inputs=[x_input, t_input], outputs=output_u)


@tf.function
def custom_loss(inputs, model):
    xyt = inputs
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    # zeros = tf.zeros_like(x)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        tape.watch(x)
        tape.watch(y)
        with tf.GradientTape(persistent=True) as tape2:            
            tape2.watch(x)
            tape2.watch(y)
            with tf.GradientTape(persistent=True) as tape3:
                tape3.watch(t)
                tape3.watch(x)
                tape3.watch(y)
                u_model = model(tf.concat([x,y,t], axis=1))
            u_x = tape3.gradient(u_model, x)
            u_y = tape3.gradient(u_model, y)
            u_t = tape3.gradient(u_model, t)
        u_xx = tape2.gradient(u_x, x)
        u_xy = tape2.gradient(u_x, y)
    u_xxx = tape.gradient(u_xx, x)
    u_xyy = tape.gradient(u_xy, y)
    del tape, tape2, tape3
      
    
    # v = -nu*u_x
    # phi_t = Vprime(u_model) - nu*u_xx - Vprime(u_model_0) + nu*u_0_xx
    # w = -nu * u_xx + phi_t/2. - Vprime(u_model)
    
    # Compute the components of loss function
    pde_loss = tf.reduce_mean((u_t + u_model * u_x + u_xxx + u_xyy) ** 2)
    
    # x_ic = tf.random.uniform((Nx*Ny,1), xMin, xMax)
    # y_ic = tf.random.uniform((Nx*Ny,1), yMin, yMax)
    x_ic = tf.expand_dims(tf.linspace(xMin, xMax, Nx*Ny), axis=-1)  # For grid sampling
    y_ic = tf.expand_dims(tf.linspace(yMin, yMax, Nx*Ny), axis=-1)  # For grid sampling
    t_ic = tf.zeros_like(x_ic)
    u_ic = u_0(x_ic, y_ic)  # Initial condition
    t_ic = tf.zeros_like(x_ic)  # t=0 for initial condition
    u_ic_pred = model(tf.concat([x_ic, y_ic, t_ic], axis=1))  # Predicted initial condition
    data_fitting_loss_0 = tf.reduce_mean((u_ic_pred - u_ic) ** 2)
    data_fitting_loss_l_r = periodic_boundary_conditions(model)

    # Combine the components of the loss functions
    # loss, loss_type = linear_loss_function([pde_loss, data_fitting_loss_0, data_fitting_loss_l_r], tf.exp(lambdas))
    # loss, loss_type = linear_loss_function([pde_loss, data_fitting_loss_0, data_fitting_loss_l_r], lambdas)
    # loss, loss_type = chebyshev_loss_function([pde_loss, data_fitting_loss_0, data_fitting_loss_l_r], tf.exp(lambdas))
    # loss, loss_type = chebyshev_loss_function([pde_loss, data_fitting_loss_0, data_fitting_loss_l_r], lambdas)
    # loss, loss_type = smooth_chebyshev_loss_function(.1, [pde_loss, data_fitting_loss_0, data_fitting_loss_l_r], lambdas)
    loss, loss_type = augmentedChebyshev_loss_function([pde_loss, data_fitting_loss_0, data_fitting_loss_l_r], lambdas)
   
    # S_loss = S(u_model, v, w)
    H_loss = H(tf.reshape(u_model, shape=[Nx, Ny, Nt]), tf.reshape(u_x, shape=[Nx, Ny, Nt]), tf.reshape(u_y, shape=[Nx, Ny, Nt]))
    # beta = 1e-3
    # data_fitting_loss = loss = beta*tf.math.log(tf.math.exp(data_fitting_loss_weight_0 * data_fitting_loss_0 / beta) 
    #                                             + tf.math.exp(data_fitting_loss_weight_l * data_fitting_loss_l / beta) 
    #                                             + tf.math.exp(data_fitting_loss_weight_r * data_fitting_loss_r / beta))
    # loss = beta*tf.math.log(tf.math.exp(pde_loss_weight * pde_loss / beta) 
    #                         + tf.math.exp(data_fitting_loss_weight_0 * data_fitting_loss_0 / beta) 
    #                         + tf.math.exp(data_fitting_loss_weight_l * data_fitting_loss_l / beta) 
    #                         + tf.math.exp(data_fitting_loss_weight_r * data_fitting_loss_r / beta))
    # data_fitting_loss = tf.math.reduce_max(tf.constant([data_fitting_loss_weight_0 * data_fitting_loss_0, 
    #                                                     data_fitting_loss_weight_l * data_fitting_loss_l, 
    #                                                     data_fitting_loss_weight_r * data_fitting_loss_r]))
    # loss = tf.math.reduce_max(tf.constant([pde_loss_weight * pde_loss, 
    #                                        data_fitting_loss_weight_0 * data_fitting_loss_0, 
    #                                        data_fitting_loss_weight_l * data_fitting_loss_l, 
    #                                        data_fitting_loss_weight_r * data_fitting_loss_r]))

    return loss, loss_type, pde_loss, data_fitting_loss_0, data_fitting_loss_l_r, H_loss#, S_loss


# Create the PINN model
model = PINNModel()
model.summary()

epochs = 500  # 5000  # 1000
# # Compile the model
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#               loss=lambda y_true, y_pred: custom_loss([x_train, t_train, theta_train], model)[1])

# Create the optimizer with a smaller learning rate
# learning_rate = 1e-3  # 1e-4
# learning_rate_type = 'constant'
# learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([10, 100], [1e-1, 5e-2, 1e-2])  #OK
# learning_rate = tf.keras.optimizers.schedules.PiecewiseConstantDecay([100, 300], [1e-2, 1e-3, 1e-4])
learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1e-2,
    decay_steps=epochs,
    end_learning_rate=1e-4,
    power=3.,
    cycle=False,
    name= 'PolynomialDecay'
)
# learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-3,
#     decay_steps=50, # 100
#     decay_rate=0.9,
#     staircase=False,
#     name='ExponentialDecay'
# )
# learning_rate = tf.keras.optimizers.schedules.CosineDecay(
#     initial_learning_rate=1e-3,
#     decay_steps=1000,
#     alpha=0.0,
#     warmup_target=None,
#     warmup_steps=0,
#     name='CosineDecay'
# )
learning_rate_type = learning_rate.name

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
# S_losses_min = []
# S_losses_max = []
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
inputs = xyt_train
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
        # param_values.append((trainable[-1]).numpy())
        # delta_gradients.append((gradients[-1]).numpy())
        # S_loss_min = tf.reduce_min(S_loss)
        # S_loss_max = tf.reduce_max(S_loss)
        # S_losses_min.append(S_loss_min.numpy())
        # S_losses_max.append(S_loss_max.numpy())
        H_loss_min = tf.reduce_min(H_loss)
        H_loss_max = tf.reduce_max(H_loss)
        H_losses_min.append(H_loss_min.numpy())
        H_losses_max.append(H_loss_max.numpy())
        H_loss_mean = tf.reduce_mean(H_loss)
        H_loss_std = tf.math.reduce_std(H_loss)
        H_losses_mean.append(H_loss_mean.numpy())
        H_losses_std.append(H_loss_std.numpy())
        # lambdas_values.append((trainable[-1]).numpy())
        
        H0 = H_loss[0].numpy()
        Hf = H_loss[-1].numpy()
        H_abs_error = tf.abs(Hf - H0)
        H_losses_abs_error.append(H_abs_error.numpy())
        H_rel_error = H_abs_error / tf.abs((H0 + 1e-16))
        H_losses_rel_error.append(H_rel_error.numpy())

        # # Print S_loss, H_loss
        # print(f"S_loss at epoch {epoch + 1}: {S_loss.numpy()}")
        # print(f"H_loss at epoch {epoch + 1}: {H_loss.numpy()}")
        
        if len(losses) > 1 and not lambdas.trainable:# and False:
        # SoftAdaptive weights update
            # num1 = tf.math.exp(pde_losses[-1] - pde_losses[-2])
            # num2 = tf.math.exp(data_fitting_losses_0[-1] - data_fitting_losses_0[-2])
            # num3 = tf.math.exp(data_fitting_losses_l_r[-1] - data_fitting_losses_l_r[-2])  
            num = tf.nn.softmax([pde_losses[-1] - pde_losses[-2], data_fitting_losses_0[-1] - data_fitting_losses_0[-2], data_fitting_losses_l_r[-1] - data_fitting_losses_l_r[-2]])
            num1 = num[0]
            num2 = num[1]
            num3 = num[2]
            den  = num1 + num2 + num3

            new_lambdas = tf.stack([num1 / den, num2 / den, num3 / den])
            lambdas.assign(new_lambdas)
            # lambdas_values.append((lambdas).numpy())
        
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
print(f"Hamiltonian absolute error: {H_abs_error.numpy()}")
print(f"Hamiltonian relative error: {H_rel_error.numpy()}")
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
plt.semilogy(data_fitting_losses_l_r, label='Periodic Boundary Conditions Loss')
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

# # Evaluate the function
# x_eval = np.linspace(x_train[0].numpy(), x_train[-1].numpy(), 100).reshape((-1, 1)).astype(np.float32)
# y_eval = np.linspace(y_train[0].numpy(), y_train[-1].numpy(), 100).reshape((-1, 1)).astype(np.float32)
# t_eval = np.linspace(t_train[0].numpy(), t_train[-1].numpy(), 100).reshape((-1, 1)).astype(np.float32)
# inputs_eval = [x_eval, y_eval, t_eval]

# # Plot the parameters over epochs
# plt.plot(S_losses_min, label='S_loss_min')
# plt.plot(S_losses_max, label='S_loss_max')
# plt.xlabel('Epoch')
# plt.ylabel('Multisymplectic Constant')
# plt.title('Multisymplectic Constant over epochs')
# plt.legend()
# plt.grid()
# 
# if save_fig:    
#     save_fig_string = generate_save_fig_string('S_loss', epochs, learning_rate_type, loss_type)
#     # save png
#     plt.savefig(save_fig_string, dpi=300)
#     # # save pdf
#     # plt.savefig('../results/' + 'S_loss.pdf', dpi=300)
    

# Plot the Hamiltonian over epochs
plt.plot(H_losses_min, label='H_loss_min')
plt.plot(H_losses_max, label='H_loss_max')
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
H_losses_abs_error = np.array(H_losses_abs_error)
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

df.to_csv('./results/2D/training_history.csv', index=False)
# from mpl_toolkits.mplot3d import Axes3D

# # Set up meshgrid
# N = 600
# tspace = np.linspace(0, 2, N + 1)
# xspace = np.linspace(0, 2, N + 1)
# yspace = np.linspace(0, 2, N + 1)
# T, X , Y= np.meshgrid(tspace, xspace, yspace)
# XYTgrid = np.vstack([X.flatten(),Y.flatten(),T.flatten()]).T

# # Determine predictions of u(t, x)
# u_pred = model(tf.cast(XYTgrid,DTYPE))

# # Reshape upred
# U = u_pred.numpy().reshape(N+1,N+1,N+1)

# # Surface plot of solution u(t,x)
# fig = plt.figure(figsize=(9,6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, U, cmap='viridis')
# ax.view_init(35,35)
# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')
# ax.set_zlabel('$u_\\theta(x,y,t)$')
# ax.set_title('Solution to KdV equation')
# if save_fig:
#     save_fig_string = generate_save_fig_string('sol', epochs, learning_rate_type, loss_type)
#     # save png
#     plt.savefig(save_fig_string, dpi=300)
#     # # save pdf
#     # plt.savefig('../results/' + 'solution.pdf', dpi=300)

# # Extract the components of lambdas over epochs
# lambda_1 = [l[0] for l in lambdas_values]
# lambda_2 = [l[1] for l in lambdas_values]
# lambda_3 = [l[2] for l in lambdas_values]

# # Plot the components of lambdas
# plt.figure(figsize=(10, 6))
# plt.plot(lambda_1, label='$\lambda_1$', color='r')
# plt.plot(lambda_2, label='$\lambda_2$', color='g')
# plt.plot(lambda_3, label='$\lambda_3$', color='b')
# plt.xlabel('Epochs')
# plt.ylabel('Weights Values')
# plt.title('Evolution of weight components over training')
# plt.legend()
# plt.grid()
# 

# # Save the plot if required
# if save_fig:
#     save_fig_string = generate_save_fig_string('lambdas', epochs, learning_rate_type, loss_type)
#     plt.savefig(save_fig_string, dpi=300)

