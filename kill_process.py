import psutil
import sys
# This function kills all the child processes associated with the parent process sent as function argument. 
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

pid = int(sys.argv[1])

kill(pid)
