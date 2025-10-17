from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MSG_MB = 50
MSG_BYTES = MSG_MB * 1024 * 1024

if size < 2:
    if rank == 0:
        print("Run with at least 2 processes")
    sys.exit(0)

msg = bytearray(MSG_BYTES)
dest = (rank + 1) % size
source = (rank - 1) % size

# Measure time
start_time = MPI.Wtime()

# Reordering to avoid deadlock: recv first, send after
if rank % 2 == 0:
    # Even processes send first
    comm.send(msg, dest=dest)
    data = comm.recv(source=source)
else:
    # Odd processes receive first
    data = comm.recv(source=source)
    comm.send(msg, dest=dest)

end_time = MPI.Wtime()

print(f"[Process {rank}] communication completed")
print(f"[Process {rank}] Total time: {end_time - start_time:.6f} s")
sys.stdout.flush()

MPI.Finalize()
