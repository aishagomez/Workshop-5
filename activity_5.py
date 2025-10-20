from mpi4py import MPI
import numpy as np
import time
import os

# ========================================
# Manual multiplication function
# ========================================
def manual_matrix_multiply(A, B):
    n_rows_A, n_cols_A = A.shape
    n_rows_B, n_cols_B = B.shape

    # Create result matrix filled with zeros
    C = np.zeros((n_rows_A, n_cols_B))

    # Classical triple-loop multiplication
    for i in range(n_rows_A):
        for j in range(n_cols_B):
            suma = 0.0
            for k in range(n_cols_A):
                suma += A[i, k] * B[k, j]
            C[i, j] = suma
    return C


# ========================================
# MPI initialization
# ========================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Matrix size
n = 500

# Only process 0 generates the matrices
if rank == 0:
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    start_time = MPI.Wtime()
else:
    A = None
    B = None
    start_time = None

# Broadcast matrix B to all processes
B = comm.bcast(B if rank == 0 else np.empty((n, n)), root=0)

# Split A into blocks of rows
rows_per_process = n // size
remainder = n % size

if rank < remainder:
    local_rows = rows_per_process + 1
    start_row = rank * local_rows
else:
    start_row = rank * rows_per_process + remainder
    local_rows = rows_per_process

# Process 0 distributes the blocks of A
if rank == 0:
    chunks = []
    start = 0
    for i in range(size):
        end = start + rows_per_process + (1 if i < remainder else 0)
        chunks.append(A[start:end, :])
        start = end
else:
    chunks = None

# Send each block to its process
local_A = comm.scatter(chunks, root=0)

# ========================================
# Local multiplication (manual)
# ========================================
local_C = manual_matrix_multiply(local_A, B)

# Gather results
C = comm.gather(local_C, root=0)

# Only process 0 assembles and measures time
if rank == 0:
    C = np.vstack(C)
    end_time = MPI.Wtime()
    total_time = end_time - start_time
    print(f"\nProcesses: {size}, Total time: {total_time:.6f} seconds")
