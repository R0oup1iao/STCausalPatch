import os
import sys
from os.path import join as opj
from os.path import dirname as opd
from os.path import basename as opb
from os.path import splitext as ops
sys.path.append(opj(os.getcwd(), "../"))
sys.path.append(os.getcwd())

import csv
import tqdm
import torch
import scipy
from .generate_data_mod import generate_random_contemp_model, generate_nonlinear_contemp_timeseries
import numpy as np
from scipy.integrate import odeint
from einops import rearrange


######################################
# Function for loading input data 
######################################
def loadTrainingData(inputDataFilePath, device):

    # Load and parse input data (create batch data)
    inpData = torch.load(inputDataFilePath)
    Xtrain = torch.zeros(inpData['TsData'].shape[1], inpData['TsData'].shape[0], requires_grad = False, device=device)
    Xtrain1 = inpData['TsData'].t()
    Xtrain.data[:,:] = Xtrain1.data[:,:]

    return Xtrain

#######################################################
# Function for reading ground truth network from file 
#######################################################
def loadTrueNetwork(inputFilePath, networkSize):

    with open(inputFilePath) as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')
        numrows = 0    
        for row in reader:
            numrows = numrows + 1

    network = np.zeros((numrows,2),dtype=np.int16)
    with open(inputFilePath) as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')
        rowcounter = 0
        for row in reader:
            network[rowcounter][0] = int(row[0][1:])
            network[rowcounter][1] = int(row[1][1:])
            rowcounter = rowcounter + 1 

    Gtrue = np.zeros((networkSize,networkSize), dtype=np.int16)
    for row in range(0,len(network),1):
        Gtrue[network[row][1]-1][network[row][0]-1] = 1   
    
    return Gtrue


def load_dream_data(dataset_id):
    device = "cpu"

    if(dataset_id == 0):
        InputDataFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Ecoli1.pt"
        RefNetworkFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Ecoli1.tsv"
    elif(dataset_id == 1):
        InputDataFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Ecoli2.pt"
        RefNetworkFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Ecoli2.tsv"
    elif(dataset_id == 2):
        InputDataFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Yeast1.pt"
        RefNetworkFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Yeast1.tsv"
    elif(dataset_id == 3):
        InputDataFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Yeast2.pt"
        RefNetworkFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Yeast2.tsv"
    elif(dataset_id == 4):
        InputDataFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/Dream3TensorData/Size100Yeast3.pt"
        RefNetworkFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize100-Yeast3.tsv"
    elif(dataset_id == 5):
        InputDataFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/Dream3TensorData/Size10Ecoli1.pt"
        RefNetworkFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize10-Ecoli1.tsv"
    elif(dataset_id == 6):
        InputDataFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/Dream3TensorData/Size10Ecoli2.pt"
        RefNetworkFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize10-Ecoli2.tsv"
    elif(dataset_id == 7):
        InputDataFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/Dream3TensorData/Size10Yeast1.pt"
        RefNetworkFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize10-Yeast1.tsv"
    elif(dataset_id == 8):
        InputDataFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/Dream3TensorData/Size10Yeast2.pt"
        RefNetworkFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize10-Yeast2.tsv"
    elif(dataset_id == 9):
        InputDataFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/Dream3TensorData/Size10Yeast3.pt"
        RefNetworkFilePath = "causal_discov_sota/SRU_for_GCI/data/dream3/TrueGeneNetworks/InSilicoSize10-Yeast3.tsv"
    else:
        print("Error while loading gene training data")    

    Xtrain = loadTrainingData(InputDataFilePath, device)
    n = Xtrain.shape[0]
    Gref = loadTrueNetwork(RefNetworkFilePath, n)
    
    Xtrain = Xtrain.numpy().T
    # Gref = Gref.T
    
    return Xtrain, Gref



def links_to_matrix(links):
    N = len(links)
    cm = np.zeros([N, N])
    for i, effect_node in links.items():
        for (j, _), _, _ in effect_node:
            cm[i, j] += 1
    return cm


class noise_model:
    def __init__(self, sigma=1, seed=0):
        self.random_state = np.random.RandomState(seed)
        self.sigma = sigma

    def gaussian(self, T):
        # Get zero-mean unit variance gaussian distribution
        return self.sigma*self.random_state.randn(T)

    def weibull(self, T):
        # Get zero-mean sigma variance weibull distribution
        a = 2
        mean = scipy.special.gamma(1./a + 1)
        variance = scipy.special.gamma(
            2./a + 1) - scipy.special.gamma(1./a + 1)**2
        return self.sigma*(self.random_state.weibull(a=a, size=T) - mean)/np.sqrt(variance)

    def uniform(self, T):
        # Get zero-mean sigma variance uniform distribution
        mean = 0.5
        variance = 1./12.
        return self.sigma*(self.random_state.uniform(size=T) - mean)/np.sqrt(variance)


def lin_f(x): return x
def f2(x): return (x + 5. * x**2 * np.exp(-x**2 / 20.))


def simulate_random_var(seed, T, N, L, coef=[0.2, 0.8], auto_corr=[0.4,0.9], tau_max=5, noise_sigma=[0.01, 0.01]):

    if True:
        coupling_funcs = [lin_f]
        noise_types = ['gaussian']  # , 'weibull', 'uniform']
        # noise_sigma = (0.1, 0.3)

    couplings = list(np.arange(coef[0], coef[1]+1e-5, coef[2]))
    couplings += [-c for c in couplings]

    # auto_deps = list(np.arange(max(0., auto_corr-0.6), auto_corr+0.01, 0.05))
    auto_deps = list(np.arange(auto_corr[0], auto_corr[1]+1e-5, auto_corr[2]))

    # Models may be non-stationary. Hence, we iterate over a number of seeds
    # to find a stationary one regarding network topology, noises, etc

    ir = 0
    model_seed = seed
    while True:
        ir += 1
        # np.random.seed(model_seed)
        random_state = np.random.RandomState(model_seed)

        links = generate_random_contemp_model(
            N=N, L=L,
            coupling_coeffs=couplings,
            coupling_funcs=coupling_funcs,
            auto_coeffs=auto_deps,
            tau_max=tau_max,
            contemp_fraction=0.,
            # num_trials=1000,
            random_state=random_state)

        noises = []
        for j in links:
            noise_type = random_state.choice(noise_types)
            sigmas = list(np.arange(noise_sigma[0], noise_sigma[1]+1e-5, noise_sigma[2]))
            sigma = random_state.choice(sigmas)
            # sigma = noise_sigma[0] + (noise_sigma[1]-noise_sigma[0])*random_state.rand()
            noises.append(getattr(noise_model(sigma=sigma, seed=seed), noise_type))

        data_all_check, nonstationary = generate_nonlinear_contemp_timeseries(
            links=links, T=100, noises=noises, random_state=random_state)

        # If the model is stationary, break the loop
        if not nonstationary:
            data, nonstationary_full = generate_nonlinear_contemp_timeseries(
                links=links, T=T, noises=noises, random_state=random_state)
            if not nonstationary_full:
                break
        else:
            print("Trial %d: Not a stationary model" % ir)
            model_seed += 10000

    cm = links_to_matrix(links)
    return data, cm


def simulate_var_from_links(links, T, seed=0, noise_sigma=[0.1, 0.2], noise_type="gaussian", func_name="lin_f"):
    """
    links_coeffs = {0: [((0, -1), 0.7), ((1, -1), -0.8)],
                    1: [((1, -1), 0.8), ((3, -1), 0.8)],
                    2: [((2, -1), 0.5), ((1, -2), 0.5), ((3, -3), 0.6)],
                    3: [((3, -1), 0.4)],
                    }
    """
    def get_func(func_name):
        if func_name == "lin_f":
            return lin_f
        else:
            raise NotImplementedError

    random_state = np.random.RandomState(seed)
    noises = []

    new_links = {}
    for j in range(len(links)):
        sigma = noise_sigma[0] + \
            (noise_sigma[1]-noise_sigma[0])*random_state.rand()
        noises.append(getattr(noise_model(sigma=sigma, seed=seed), noise_type))
        new_links[j] = []
        for props in links[j]:
            new_links[j].append(
                (tuple(props[0:2]), props[2], get_func(props[3]),))
    data, nonstationary = generate_nonlinear_contemp_timeseries(
        links=new_links, T=T, noises=noises, random_state=random_state)
    if nonstationary:
        print("Model nonstationay!")

    cm = links_to_matrix(new_links)
    return data, cm


def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        # print(f"Nonstationary, beta={str(beta):s}, max_eig={max_eig:.4f}")
        return make_var_stationary((beta / max_eig) * 0.7, radius)
    else:
        # print(f"Stationary, beta={str(beta):s}")
        return beta


def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, auto_corr=3.0, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * auto_corr

    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)

    # Generate data.
    burn_in = 100
    errors = np.random.normal(loc=0, scale=sd, size=(p, T + burn_in))
    X = np.ones((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t-lag):t].flatten(order='F'))
        X[:, t] += errors[:, t-1]
        
    data = X.T[burn_in:, :]
    return data, beta, GC





def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

    return dxdt


def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,
                       seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:, :], GC

import numpy as np
from scipy.integrate import odeint

def lorenz_patch(x, t, F):
    """
    Lorenz-96 style dynamics but for patch-level state vector x (length = num_patches).
    """
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F
    return dxdt


def simulate_lorenz_96_patches(p, T, patch_size,
                               F=10.0, delta_t=0.1, sd_obs=0.1,
                               sd_internal=0.0, burn_in=1000, seed=0,
                               agg_method="mean"):
    """
    Simulate Lorenz-96 dynamics at patch level, then expand back to node level.

    Parameters
    ----------
    p : int
        Total number of nodes.
    T : int
        Number of time steps to return (after burn-in).
    patch_size : int
        Number of nodes in each patch. Last patch may be smaller if p % patch_size != 0.
    F : float
        Forcing term for Lorenz dynamics.
    delta_t : float
        Time step size (kept for compatibility with odeint time vector length).
    sd_obs : float
        Observation noise added to node-level output (after expansion).
    sd_internal : float
        Optional small internal noise within patch to give node-level diversity.
    burn_in : int
        Number of initial steps to discard.
    seed : int or None
        Random seed.
    agg_method : str
        How to aggregate nodes to a patch initial/value if needed; currently kept for interface.
        Options: "mean" or "sum" (only affects initial patch state here).

    Returns
    -------
    X_nodes : ndarray, shape (T, p)
        Node-level time series (after burn-in) where each node's value is the patch value
        plus optional internal noise and observation noise.
    GC_patch : ndarray, shape (num_patches, num_patches)
        Ground-truth (patch-level) causal adjacency (1 for self, ±1 neighbors similar to original).
    patch_assignments : list of tuples
        List of (start_idx, end_idx) for each patch (inclusive start, exclusive end).
    X_patches : ndarray, shape (T, num_patches)
        The simulated patch-level time series (after burn-in).
    """
    if seed is not None:
        np.random.seed(seed)

    # build patch partitioning
    num_patches = int(np.ceil(p / patch_size))
    patch_assignments = []
    for i in range(num_patches):
        start = i * patch_size
        end = min((i + 1) * patch_size, p)
        patch_assignments.append((start, end))

    # initial patch-level state: small random perturbation
    x0_patch = np.random.normal(scale=0.01, size=num_patches)

    # integrate patch-level ODE
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X_patch_full = odeint(lorenz_patch, x0_patch, t, args=(F,))  # shape (T+burn_in, num_patches)

    # optionally add small process noise to patches (kept deterministic here; observation noise added below)

    # build ground-truth patch-level GC adjacency
    GC_patch = np.zeros((num_patches, num_patches), dtype=int)
    for i in range(num_patches):
        GC_patch[i, i] = 1
        GC_patch[i, (i + 1) % num_patches] = 1
        GC_patch[i, (i - 1) % num_patches] = 1
        GC_patch[i, (i - 2) % num_patches] = 1

    # discard burn-in
    X_patches = X_patch_full[burn_in:, :]  # shape (T, num_patches)

    # expand patch time series to node-level
    X_nodes = np.zeros((T, p))
    for t_idx in range(T):
        for patch_i, (start, end) in enumerate(patch_assignments):
            # base patch value at this time
            val = X_patches[t_idx, patch_i]

            # create internal variation if requested
            if sd_internal > 0:
                internal_noise = np.random.normal(scale=sd_internal, size=(end - start))
                node_vals = val + internal_noise
            else:
                node_vals = np.full((end - start,), val)

            X_nodes[t_idx, start:end] = node_vals

    # add observation noise to node-level series
    if sd_obs > 0:
        X_nodes += np.random.normal(scale=sd_obs, size=X_nodes.shape)
        
    # build node-level GC for full compatibility
    GC_node = np.zeros((p, p), dtype=int)
    for i, (s_i, e_i) in enumerate(patch_assignments):
        for j, (s_j, e_j) in enumerate(patch_assignments):
            if GC_patch[i, j] == 1:
                GC_node[s_i:e_i, s_j:e_j] = 1

    # backward-compatible outputs
    data = X_nodes
    true_cm = GC_node

    return data, true_cm

def prepross_data(data):
    T, N, D = data.shape
    new_data = np.zeros_like(data, dtype=float)
    for i in range(N):
        node = data[:,i,:]
        new_data[:,i,:] = (node - np.mean(node)) / np.std(node)
        
    return new_data
         