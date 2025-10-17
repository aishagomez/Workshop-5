from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initial number defined only in process 0
value = None
start_time = MPI.Wtime()  # start the timer

if rank == 0:
    value = 0  # initial number
    print(f"Process {rank} starts the chain with value {value}")
    comm.send(value, dest=rank + 1)  # send to the next process

elif rank < size - 1:
    value = comm.recv(source=rank - 1)  # receive from previous process
    value += 1  # increment the value
    print(f"Process {rank} received {value - 1}, incremented to {value}, and sends it to process {rank + 1}")
    comm.send(value, dest=rank + 1)  # send to the next

else:
    value = comm.recv(source=rank - 1)  # last process receives
    value += 1
    end_time = MPI.Wtime()  # end the timer
    print(f"Process {rank} received {value - 1}, incremented to {value}. Final value of the chain.")
    print(f"Total time: {end_time - start_time:.6f} seconds")

MPI.Finalize()
