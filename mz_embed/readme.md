## Instructions for setting up the VM for pretraining with GPU

When using an Nvidia T4 GPU, I used GCP's Deep Learning VM image:
- `c0-deeplearning-common-gpu-v20240613-debian-11-py310`
    -  Debian 11, Python 3.10. With CUDA 11.8 preinstalled
    - When you create the VM and login, it will ask you to install the Nvidia driver. Do it:
        - "This VM requires Nvidia drivers to function correctly.   Installation takes ~1 minute.
        Would you like to install the Nvidia driver? [y/n]"
        - In the terminal type: `nvidia-smi` to check if the GPU is working.
        - If you get an error, try restarting the VM.
When using an Intel CPU without any gpu, use the GCP's deep learning VM image
    `c0-deeplearning-common-cpu-v20240708-debian-11`
    -Debian 11, Python 3.10, Intel MKL. With Intel optimized NumPy, SciPy, and scikit-learn preinstalled.

- Once the gpu has been set up, run the following command to install some dependencies:
    - `sudo apt-get install -y git screen htop build-essential pkg-config libmariadb-dev-compat wget vim`
- Update the settings for screen:
    - `echo "escape ^Jj" > .screenrc`

- Clone or download the repo and enter the directory:
    - `git clone git@bitbucket.org:revivemed/mz_embed_engine.git`
    - `cd mz_embed_engine`
    
- Install the requirements
    - `pip install -r requirements.txt` or `pip install -r requirements_4.txt`
- move to mz_embed folder and run to make a package, if build not available: pip install build
    - 'cd mz_embed'
    - 'python -m build'
    - 'pip install -e .'

