"""
MTF reference implementation matching the Zig algorithm in src/TensorToImage/mtf/.

Usage:
    python3 tests/TensorToImage/mtf_reference.py

Prints bins, transition matrix, and flattened MTF for the reference test input.
"""
import numpy as np


def quantile_bins(input_arr, q):
    """
    Assigns each value to a bin in [0, q-1].
    Interior boundaries at sorted index k*n//q for k = 1..q-1.
    Matches the Zig quantileBins function exactly.
    """
    n = len(input_arr)
    sorted_arr = np.sort(input_arr)
    result = []
    for val in input_arr:
        b = 0
        for k in range(1, q):
            edge_idx = (k * n) // q
            if val >= sorted_arr[edge_idx]:
                b += 1
        result.append(min(b, q - 1))
    return result


def transition_matrix(bins, q):
    """
    Builds row-stochastic Q×Q transition matrix.
    Matches the Zig transitionMatrix function exactly.
    """
    W = np.zeros((q, q), dtype=np.float32)
    for t in range(len(bins) - 1):
        W[bins[t]][bins[t + 1]] += 1.0
    for r in range(q):
        row_sum = W[r].sum()
        if row_sum > 0:
            W[r] /= row_sum
    return W


def mtf(input_arr, q):
    """
    Builds the N×N Markov Transition Field.
    M[i][j] = W[bin[i]][bin[j]].
    Matches the Zig lean_mtf function exactly.
    """
    bins = quantile_bins(input_arr, q)
    W = transition_matrix(bins, q)
    n = len(input_arr)
    M = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            M[i][j] = W[bins[i]][bins[j]]
    return M


if __name__ == "__main__":
    input_arr = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.6], dtype=np.float32)
    q = 4

    bins = quantile_bins(input_arr, q)
    print("bins:", bins)

    W = transition_matrix(bins, q)
    print("transition matrix (row-major):")
    print(W)

    M = mtf(input_arr, q)
    print("MTF (row-major flat):")
    print(M.flatten())
