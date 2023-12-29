import subprocess
import os

os.chdir('../../../')

# Command to start Jupyter Notebook
command = ["jupyter", "notebook"]

# Start the Jupyter Notebook
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Print the output and error (if any)
stdout, stderr = process.communicate()

if process.returncode == 0:
    print("Jupyter Notebook started successfully")
    print(stdout.decode())
else:
    print("Error in starting Jupyter Notebook")
    print(stderr.decode())

