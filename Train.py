import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--node", type=int, help="NODEs")
parser.add_argument("--GPU_index", type=int, help="GPU_idx")
parser.add_argument("--alpha1", type=float, help="First augements")
parser.add_argument("--alpha2", type=float, help="Second augements")
parser.add_argument("--alpha3", type=float, help="Third augements")
parser.add_argument("--alpha4", type=float, help="Fourth augements")
parser.add_argument("--alpha5", type=float, help="Fifth augements")
parser.add_argument("--alpha6", type=float, help="Sixth augements")

args = parser.parse_args()

NODE = args.node
GPU_index = args.GPU_index

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_index)

# From here down is your code
import jax
from jax.lib import xla_bridge

import jax
from jax.example_libraries import stax, optimizers
import jax.numpy as jnp
from jax import grad, vmap, random, jit, lax

from jax.config import config
import jax.numpy as jnp

from jax.nn.initializers import normal, zeros, ones

import pandas as pd

# config.update("jax_enable_x64", True)

import pandas as pd

print(os.getcwd())
from load import load_data

mc_alpha = args.alpha4
n_seq = int(args.alpha2)
n_seq_mc = 1

num_train = 49
num_test = 1
num_epochs = 500000

learning_rate = 1e-3
batch_size = int(args.alpha1)
# In sptial space
xmin, xmax = 0, 1

nx_save = 1535

dx = (xmax - xmin) / (nx_save)
# In time space
dt = args.alpha3

nt_step_train = 400
nt_step_test = 400

neurons = nx_save+1

layer = int(args.alpha5)
neurons_hidden = int(args.alpha6)

filename = 'Wave_eq_2_HighRes_' + '_alpha_' + str(mc_alpha) + '_lr_' + str(learning_rate) + '_batch_' + str(batch_size) + '_nseq_' + str(n_seq) + '_n_seq_mc_' + str(n_seq_mc) + '_dt_' + str(dt) + '_layers_' + str(layer) + '_neurons_' + str(neurons_hidden)
import wandb
wandb.init(project="2Dflow", entity="colenockolds", name = filename)
wandb.config.problem = 'Wave_eq'
wandb.config.mc_alpha = mc_alpha
wandb.config.learning_rate = learning_rate
wandb.config.batch_size = batch_size
wandb.config.n_seq = n_seq
wandb.config.network = 'Linear'

# 1. Loading Data
XY_mat, EToV, Train_data, Test_data = load_data()

#! 2. Building up a neural network

if layer == 1:
    forward_pass_int, forward_pass = stax.serial(
        stax.Dense(neurons, W_init=normal(0.02), b_init=zeros),
    )

if layer == 2:
    forward_pass_int, forward_pass = stax.serial(
        stax.Dense(neurons_hidden, W_init=normal(0.02), b_init=zeros),
        stax.Relu,
        stax.Dense(neurons, W_init=normal(0.02), b_init=zeros)
    )

if layer == 3:
    forward_pass_int, forward_pass = stax.serial(
        stax.Dense(neurons_hidden, W_init=normal(0.02), b_init=zeros),
        stax.Relu,
        stax.Dense(neurons_hidden, W_init=normal(0.02), b_init=zeros),
        stax.Relu,
        stax.Dense(neurons, W_init=normal(0.02), b_init=zeros)
    )

_, int_params = forward_pass_int(random.PRNGKey(0), (nx_save+1,))
W1, b1 = int_params[0]
print(W1.shape, b1.shape)

for w in int_params:
    if w:
        w, b = w
        print("Weights : {}, Biases : {}".format(w.shape, b.shape))

print('=' * 20 + ' >> Success!')

#! 3. Forward solver (single time step)
dudx = jnp.roll(1 * jnp.diag(jnp.ones((nx_save + 1,1)).squeeze()), 0, axis = 0) + jnp.roll(-1 * jnp.diag(jnp.ones((nx_save + 1,1)).squeeze()), 1, axis = 0)

@jit
def single_solve_forward(u):
    u = u - dt/dx * dudx @ u
    return u

@jit
def single_forward_pass(params, u):
    u = u - dt * forward_pass(params, u)
    return u
batched_solve_forward = vmap(single_solve_forward, in_axes = 0)

# ## Loss functions and relative error/accuracy rate function
# %%
def MSE(pred, true):
    return jnp.mean(jnp.square(pred - true))

def squential_mc(i, args):
    
    loss_mc, u_mc, u_ml, params = args
    u_ml_next = single_forward_pass(params, u_ml)
    u_mc_next = single_solve_forward(u_mc)
    
    loss_mc += MSE(u_mc, u_ml_next)

    return loss_mc, u_mc_next, u_ml_next, params

def squential_ml_second_phase(i, args):
    ''' I have checked this loss function!'''

    loss_ml, loss_mc, u_ml, u_true, params = args
    
    # # This is u_mc for the current
    u_mc = single_solve_forward(u_ml)
    
    # # The forward model-constrained loss
    loss_mc, _, _, _ = lax.fori_loop(0, n_seq_mc, squential_mc, (loss_mc, u_mc, u_ml, params))

    # This is u_ml for the next step
    u_ml_next = single_forward_pass(params, u_ml)

    # The machine learning term loss
    loss_ml += MSE(u_ml, u_true[i,:])

    return loss_ml, loss_mc, u_ml_next, u_true, params


def loss_one_sample_one_time(params, u):
    loss_ml = 0
    loss_mc = 0

    # first step prediction
    u_ml = single_forward_pass(params, u[0, :])

    # for the following steps up to sequential steps n_seq
    loss_ml,loss_mc, u_ml, _, _ = lax.fori_loop(1, n_seq+1, squential_ml_second_phase, (loss_ml, loss_mc, u_ml, u, params))
    loss_ml += MSE(u_ml, u[-1, :])

    return loss_ml + mc_alpha * loss_mc

loss_one_sample_one_time_batch = vmap(loss_one_sample_one_time, in_axes = (None, 0), out_axes=0)

@jit
def transform_one_sample_data(u_one_sample):
    u_out = jnp.zeros((nt_step_train - n_seq - 1, n_seq+2, nx_save+1))
    for i in range(nt_step_train-n_seq-1):
        u_out = u_out.at[i,:,:].set(u_one_sample[i:i + n_seq + 2,:])
    return u_out

@jit
def loss_one_sample(params,u_one_sample):
    u_batch_nt = transform_one_sample_data(u_one_sample)
    return jnp.sum(loss_one_sample_one_time_batch(params, u_batch_nt))

loss_one_sample_batch = vmap(loss_one_sample, in_axes = (None, 0), out_axes=0)

@jit
def LossmcDNN(params, data):
    return jnp.sum(loss_one_sample_batch(params, data))

# ## Computing test error, predictions over all time steps
@jit
def neural_solver(params, U_test):
    
    u = U_test[0,:]
    U = jnp.zeros((nt_step_test + 1, nx_save + 1))
    U = U.at[0, :].set(u)
    
    for i in range(1, nt_step_test + 1):
        u = single_forward_pass(params, u)
        U = U.at[i, :].set(u)
        
    return U

neural_solver_batch = vmap(neural_solver, in_axes = (None, 0))

@jit
def test_acc(params, Test_set):
    return MSE(neural_solver_batch(params, Test_set), Test_set)

# %%
#! 5. Epoch loops fucntions and training settings 
def body_fun(i, args):
    opt_state, data = args
    data_batch = lax.dynamic_slice_in_dim(data, i * batch_size, batch_size)

    gradients = grad(LossmcDNN)(opt_get_params(opt_state), data_batch)

    return opt_update(i, gradients, opt_state), data

@jit
def run_epoch(opt_state, data):
    return lax.fori_loop(0, num_batches, body_fun, (opt_state, data))

def TrainModel(train_data, test_data, num_epochs, opt_state):
    
    loss_history, test_acc_history = [], []
    
    test_accuracy_min = 100
    epoch_min = 1
    
    for epoch in range(1,num_epochs+1):
        opt_state, _ = run_epoch(opt_state, train_data)
        
        train_loss = LossmcDNN(opt_get_params(opt_state), train_data)

        test_accuracy = test_acc(opt_get_params(opt_state), test_data)
        
        if test_accuracy_min >= test_accuracy:
            test_accuracy_min = test_accuracy
            epoch_min = epoch
            optimal_opt_state = opt_state

        if epoch % 1000 == 0:  # Print MSE every 1000 epochs
            print("Data_d {:d} n_seq {:d} batch {:d} lr {:.2e} loss {:.2e} TE {:.2e}  TE_min {:.2e} EPmin {:d} EP {} ".format(
                num_train, n_seq, batch_size, learning_rate, train_loss, test_accuracy, test_accuracy_min, epoch_min, epoch))

        if epoch % 1 == 0:  
            wandb.log({"Train loss": float(train_loss), "Test Error": float(test_accuracy), 'TEST MIN' : float(test_accuracy_min)})

    return optimal_opt_state

# %%
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)

opt_int, opt_update, opt_get_params = optimizers.adam(learning_rate)
opt_state = opt_int(int_params)

final_opt_state = TrainModel(Train_data, Test_data, num_epochs, opt_state)

import pickle

trained_params = optimizers.unpack_optimizer_state(final_opt_state)
pickle.dump(trained_params, open('../src/network/Best_trial_' + filename, "wb"))