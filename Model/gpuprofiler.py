import subprocess

proc = subprocess.Popen(['./scr'])
print("start process with pid %s" % proc.pid)

#proc.kill()

