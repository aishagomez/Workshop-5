from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Tamaño del mensaje en MB (ajusta si tu máquina tiene poca RAM)
MSG_MB = 50
MSG_BYTES = MSG_MB * 1024 * 1024

if size < 2:
    if rank == 0:
        print("Ejecuta con al menos 2 procesos: mpirun -n 2 python3 mpi_deadlock_forced.py")
    sys.exit(0)

# Preparar un mensaje grande para evitar buffering automático
msg = bytearray(MSG_BYTES)  # mensaje "vacío" de MSG_MB megabytes

# Destino y fuente (cadena circular)
dest = (rank + 1) % size
source = (rank - 1) % size

# Mensajes informativos (se verán antes del bloqueo)
print(f"[Proceso {rank}] preparado para ssend de {MSG_MB} MB hacia {dest}")
sys.stdout.flush()

# Intento de envío síncrono: esto bloqueará hasta que el receptor real reciba
comm.ssend(msg, dest=dest)
print(f"[Proceso {rank}] ssend completado hacia {dest} (esto probablemente NO se mostrará)")
sys.stdout.flush()

# Intento de recibir (nunca llegará si todos están bloqueados en ssend)
data = comm.recv(source=source)
print(f"[Proceso {rank}] recibió datos desde {source} (esto probablemente NO se mostrará)")
sys.stdout.flush()

MPI.Finalize()
