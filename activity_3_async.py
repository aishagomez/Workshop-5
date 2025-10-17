from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MSG_MB = 1  # Reduce to 1 KB
MSG_BYTES = MSG_MB * 1024

if size < 2:
    if rank == 0:
        print("Run with at least 2 processes")
    sys.exit(0)

msg = bytearray(MSG_BYTES)
dest = (rank + 1) % size
source = (rank - 1) % size

start_time = MPI.Wtime()

req_send = comm.isend(msg, dest=dest)
req_recv = comm.irecv(source=source)

data = req_recv.wait()
req_send.wait()

end_time = MPI.Wtime()

print(f"[Process {rank}] communication completed")
print(f"[Process {rank}] Total")
