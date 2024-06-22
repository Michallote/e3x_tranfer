# E3x ===>
import functools
import io
import os
import urllib.request

import ase
import ase.io as ase_io
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

import e3x

jax.devices()

# Disable future warnings.
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from .E3x_MLFF import MessagePassingModel as MPM

# --- Load E3x model file. ---
sGDML_model = "best_model_params_test_mode.pkl"

try:
    import pickle

    with open(sGDML_model, "rb") as file:
        model = pickle.load(file)

    print(" @ForceField: sGDML model " + sGDML_model + " loaded")
except:
    print("ERROR: Reading sGDML model " + sGDML_model + " file failed.")

# Hyperparameters
# hyperparameters = model["hyperparameters"]

# Model hyperparameters.
features = 32
max_degree = 1
num_iterations = 5
num_basis_functions = 16
cutoff = 10.0
"""hyperparameters = {
    "features": 32,
    "max_degree": 2,
    "num_iterations": 3,
    "num_basis_functions": 16,
    "cutoff": 5.0,
}

features = hyperparameters['features']
max_degree = hyperparameters['max_degree']
num_iterations = hyperparameters['num_iterations']
num_basis_functions = hyperparameters['num_basis_functions']
cutoff = hyperparameters['cutoff']"""


# --- Creates predictor ---
# Create model.
message_passing_model = MPM(
    features=features,
    max_degree=max_degree,
    num_iterations=num_iterations,
    num_basis_functions=num_basis_functions,
    cutoff=cutoff,
)
print(f" message_passing_model initialized")

atomic_numbers = jnp.array(
    [23, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14]
)
num_atoms = len(atomic_numbers)
dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(num_atoms)

# ----------> Batch
nbeads = 10

if nbeads > 1:
    batch_segments = np.repeat(np.arange(nbeads), num_atoms)
    batch_size = nbeads
    atomic_numbers = jnp.tile(atomic_numbers, batch_size)
    offsets = jnp.arange(batch_size) * num_atoms
    dst_idx = (dst_idx + offsets[:, None]).reshape(-1)
    src_idx = (src_idx + offsets[:, None]).reshape(-1)
else:
    batch_segments = None
    batch_size = None


@jax.jit
def evaluate_energies_and_forces(atomic_numbers, positions, dst_idx, src_idx):
    return message_passing_model.apply(
        model,
        atomic_numbers=atomic_numbers,
        positions=positions,
        dst_idx=dst_idx,
        src_idx=src_idx,
        batch_segments=batch_segments,
        batch_size=batch_size,
    )


print(f" evaluate_energies_and_forces defined")


def predictor(X):
    # dst_idx, src_idx = e3x.ops.sparse_pairwise_indices(len(atomic_numbers))
    energy, forces = evaluate_energies_and_forces(
        atomic_numbers=atomic_numbers,
        positions=X,
        dst_idx=dst_idx,
        src_idx=src_idx,
    )
    return energy, forces


print(f" predictor defined")

"""
Define X como un un batch de coordenadas 
"""

filename = "/home/beemoqc2/Documents/e3x/docs/source/examples/datos_modificados.npz"
dataset = np.load(filename)

X = np.array([dataset["R"][0]])


def evaluate_bulk(requests):
    """Evaluate the energy and forces."""
    R = np.array([x.reshape(-1, 3) for x in requests]).reshape(-1, 3)
    # print("R.shape", R.shape, len(atomic_numbers))

    E, F = predictor(R * 10)
    # print("E.shape", E.shape, "F.shape", F.shape)

    return E, F


E, F = evaluate_bulk(X)
