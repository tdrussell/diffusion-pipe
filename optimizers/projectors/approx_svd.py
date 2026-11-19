# Here we implement additional approximate SVD to speedup SVD computation
import numpy as np
import math
import importlib.util
import torch


# Check if fast_hadamard_transform is installed
def is_package_installed(package_name):
    return importlib.util.find_spec(package_name) is not None


if is_package_installed("fast_hadamard_transform"):
    from fast_hadamard_transform import hadamard_transform
else:
    # Default Hadamard transform function for PyTorch
    def hadamard_transform(x: torch.Tensor):
        raise NotImplementedError('Need to install "fast_hadamard_transform" first: '
                                  'https://github.com/Dao-AILab/fast-hadamard-transform')


def approximate_svd(input_matrix: torch.Tensor, subsample_size: int, seed=None):
    """
    Perform approximate Singular Value Decomposition (SVD) using Subsampled Randomized Hadamard Transform (SRHT).

    Args:
    - input_matrix (torch.Tensor): Input matrix of shape (n, d)
    - subsample_size (int): Size k of the subsampled rows for SRHT (k)

    Returns:
    - U (torch.Tensor): Left singular vectors of shape (n, k)
    """
    assert len(input_matrix.shape) == 2, "Only handles 2d tensors"
    rng = np.random if seed is None else np.random.RandomState(seed)
    n, d = input_matrix.shape
    manual_seed = rng.randint(1000000000)

    # Step 1: Compute SRHT of input
    srht_matrix = srht(input_matrix, subsample_size, manual_seed)  # (k, d)

    # Step 2: Compute SVD of SRHT matrix
    U_tilde, S, V = torch.linalg.svd(srht_matrix)  # (k, k), diag(k), (d, d)

    # Step 3: Project U_tilde back via SRHT transposed to get an approximate U for input_matrix
    U = transposed_srht(U_tilde, n, manual_seed)
    return U, S, V


def transposed_srht(input_matrix: torch.Tensor, original_size: int, manual_seed: int = None, states: dict = None, scale=True):
    """
    Compute the Transposed SRHT of a given input matrix of size (k, d) where k is the subsampled size.

    Args:
    - input_matrix (torch.Tensor): Input matrix of shape (k, d)
    - original_size (int): Original size n where the output will be (n, d)
    - manual_seed (int): Seed for efficient storage
    - states (dict): dict containing the idx and random_signs keys to avoid regenerating
    Returns:
    - srht_matrix (torch.Tensor): SRHT^T*input_matrix.
    """
    k, d = input_matrix.size()
    device = input_matrix.device
    dtype = input_matrix.dtype
    n = original_size

    assert n >= k, f"n={n} must be >= k={k} and >= d={d}."

    state_dict = get_subsample_idx_and_random_signs(n, k, device, dtype, manual_seed, states)
    idx, random_signs = state_dict['idx'], state_dict['random_signs']

    # transpose
    subsample_transposed = torch.zeros(n, k, dtype=dtype, device=device)
    subsample_transposed[idx, torch.arange(k, dtype=torch.long, device=device)] = 1

    # Compute HT of P^T A and multiply by random signs and return
    if scale:
        scale_factor = 1 / math.sqrt(n * k)
        result = scale_factor*hadamard_transform(torch.matmul(subsample_transposed, input_matrix)).mul_(random_signs.unsqueeze(0).T)
        result.to(device)
        result.to(dtype)
        return result
    return hadamard_transform(torch.matmul(subsample_transposed, input_matrix)).mul_(random_signs.unsqueeze(0).T)


def srht(input_matrix: torch.Tensor, subsample_size: int, manual_seed: int = None, states: dict = None, scale=True):
    """
    Compute the Subsampled Randomized Hadamard Transform (SRHT) of a given input matrix.

    Args:
    - input_matrix (torch.Tensor): Input matrix of shape (n, d)
    - subsample_size (int): Size of the subsampled rows
    - manual_seed (int): Seed to set to avoid storing a specific srht so we can perform inverse.
    Returns:
    - srht_matrix (torch.Tensor): SRHT*input_matrix of shape (n, subsample_size)
    """
    # need to set this for the randomness to be deterministic
    n, d = input_matrix.size()
    device = input_matrix.device
    dtype = input_matrix.dtype

    state_dict = get_subsample_idx_and_random_signs(n, subsample_size, device, dtype, manual_seed, states)
    idx, random_signs = state_dict['idx'], state_dict['random_signs']

    assert subsample_size <= n, f"Subsample size must be less than or equal to input dimension. n={n}, d={d}"

    # Compute HT of H*D and return
    if scale:
        scale_factor = 1/math.sqrt(n*subsample_size)
        result = scale_factor*hadamard_transform(torch.mul(input_matrix, random_signs.unsqueeze(0).T))[idx, :]
        result.to(device)
        result.to(dtype)
        return result
    return hadamard_transform(torch.mul(input_matrix, random_signs.unsqueeze(0).T))[idx, :]


def get_subsample_idx_and_random_signs(n, subsample_size, device, dtype, manual_seed, states):
    assert not (manual_seed is None and states is None), "must provide either the manual seed or states"

    if states is not None:  # to avoid regenerating
        assert manual_seed is None and 'idx' in states.keys() and 'random_signs' in states.keys()
        idx, random_signs = states['idx'], states['random_signs']
        assert idx.shape == (subsample_size, ) and random_signs.shape == (n, ), f"random_signs.shape={random_signs.shape}, n={n}, idx.shape={idx.shape}, subsample_size={subsample_size}."
    else:
        assert states is None
        torch.manual_seed(manual_seed)
        # Generate a random subset of rows of H*D
        idx = torch.randperm(n, device=device)[:subsample_size]
        # Construct the diagonal matrix D with +1 and -1 on diagonals
        random_signs = torch.randint(0, 2, (n,), dtype=dtype, device=device) * 2 - 1

    return {'idx': idx, 'random_signs': random_signs}


def get_subsample_idx_and_random_signs_from_matrix(input_mat, rank, manual_seed):
    n, d = input_mat.size()
    device = input_mat.device
    dtype = input_mat.dtype

    return get_subsample_idx_and_random_signs(n, rank, device, dtype, manual_seed, None)