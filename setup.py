import os
import subprocess
from setuptools import setup, find_packages

# Function to install the 'pointops' submodule
def install_pointops():
    # Get the directory in which the current script is located
    setup_dir = os.path.dirname(os.path.abspath(__file__))
    pointops_path = os.path.join(setup_dir, 'pointops')
    
    # Check if the setup.py exists for pointops
    if os.path.exists(pointops_path):
        subprocess.check_call(['pip', 'install', '-e', pointops_path])
    else:
        raise FileNotFoundError("Couldn't find pointops setup.py at {}".format(pointops_path))

# Call the function during the setup process
install_pointops()

setup(
    name='transloc4d',
    version='0.1',
    packages=find_packages(),
    install_requires=[]
)
