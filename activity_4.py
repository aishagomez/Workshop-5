from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
start_time = MPI.Wtime() # Inicio del cronómetro

# Tamaño total del array
N = 50

if rank == 0:
    data = np.arange(1, N + 1, dtype=int)
else:
    data = None

data = comm.bcast(data, root=0)

# Dividir el array en bloques para cada proceso
chunk_size = N // size
start = rank * chunk_size
if rank != size - 1:
    end = start + chunk_size 
else: 
    end = N  # último proceso puede tomar resto
sub_array = data[start:end]

# Cada proceso calcula los cuadrados de su sub_array
squares = sub_array ** 2

# Sumar localmente los cuadrados y usar reduce para obtener suma global
local_sum = np.sum(squares)
global_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)

# Gather de todos los cuadrados para el proceso 0
all_squares = comm.gather(squares, root=0)

# Tiempo final
comm.Barrier()
end_time = MPI.Wtime()
elapsed_time = end_time - start_time

if rank == 0:
    print("Array original:", data)
    print("Suma global de cuadrados:", global_sum)
    print("Todos los cuadrados por proceso:", all_squares)
    print(f"Tiempo total: {elapsed_time:.6f} s")
