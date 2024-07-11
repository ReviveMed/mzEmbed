


## Instructions for setting up the VM for pretraining with GPU

When using an Nvidia T4 GPU, I used GCP's Deep Learning VM image:
- `c0-deeplearning-common-gpu-v20240613-debian-11-py310`
- When you create the VM and login, it will ask you to install the Nvidia driver. Do it:
    - "This VM requires Nvidia drivers to function correctly.   Installation takes ~1 minute.
    Would you like to install the Nvidia driver? [y/n]"
    - In the terminal type: `nvidia-smi` to check if the GPU is working.
    - If you get an error, try restarting the VM.
- Once the gpu has been set up, run the following command to install some dependencies:
    - `sudo apt-get install -y git screen htop build-essential pkg-config libmariadb-dev-compat wget vim`
- Update the settings for screen:
    - echo "escape ^Jj" > .screenrc
- (Optional) set up an ssh connection to bitbucket:
    - `ssh -T jonaheaton@bitbucket.org`
    - If you get an error, you may need to add the bitbucket ssh key to your ssh
- Clone the repo and enter the directory:
    - `git clone jonaheaton@bitbucket.org:revivemed/mz_embed_engine.git`
    - `cd mz_embed_engine`
- (optional) Checkout the branch you want to work on
    - `git checkout <branch_name>`
- Install the requirements
    - `pip install -r requirements.txt` or `pip install -r requirements_4.txt`

