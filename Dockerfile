# Use an official Python runtime as a parent image
# FROM python:3.9-bookworm 
FROM python:3.9-slim

LABEL Description="ReviveMed Linux environment for mz_embed_engine analysis using Python (R to be added later)"
LABEL tags="revivemed-mz_embed_engine"

# Set the working directory in the container
WORKDIR /app


# the more secure method
# ARG BITBUCKET_ACCESS_KEY

# less secure
# ENV BITBUCKET_ACCESS_KEY=your_access_key


# not a super secure way to save the API token
ENV NEPTUNE_API_TOKEN=="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxMGM5ZDhiMy1kOTlhLTRlMTAtOGFlYy1hOTQzMDE1YjZlNjcifQ=="
# more secure way would be pass it at runtime:
# docker run -e API_TOKEN=your_api_token your_image_name


# Copy the current directory contents into the container at /app
# COPY . /app
# only copy the requirements_4.txt file
COPY ./requirements_4.txt /app/requirements_4.txt

# copy the Bitbucket access key
# I think these are read only on the mz-embed-engine repo
COPY ./id_ed25519 /root/.ssh/id_ed25519
COPY ./id_ed25519.pub /root/.ssh/id_ed25519.pub

RUN apt-get update

# install git
RUN apt-get install -y git
# Install useful system packages
RUN apt-get install -y screen htop
RUN apt-get install -y build-essential
RUN apt-get install -y pkg-config
RUN apt-get install -y libmariadb-dev-compat
RUN apt-get install wget
RUN apt-get update && apt-get install -y vim

RUN pip install --upgrade pip
RUN pip install mysqlclient


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements_4.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Copy the current directory contents into the container at /app
# COPY . /app

# RUN git clone https://$BITBUCKET_ACCESS_KEY@bitbucket.org/revivemed/mz_embed_engine.git
# RUN git checkout dev_jonah


# Run commands when the container launches
# CMD ["python", "./main.py"]

# Set default command to be jupyter notebook
# CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]

# If you want to make changes to your code without having to rebuild the docker image
# docker run -p 4000:80 -v $(pwd):/app your-image-name

# docker pull dockerrevivemed/mz_embed_engine:v1.2
# docker run -it -p 8080:8080 -v ~/mz_embed_engine:/app dockerrevivemed/mz_embed_engine:v1.2 /bin/bash

# git clone https://jonaheaton@bitbucket.org/revivemed/mz_embed_engine.git
# git stash
# git pull origin dev_jonah

# Add user to the docker group so you don't need sudo and can use vscode to develop:
# https://docs.docker.com/engine/install/linux-postinstall/



# git remote set-url origin git@bitbucket.org:jonaheaton/revivemed/mz_embed_engine.git
# git remote set-url origin jonaheaton@bitbucket.org:revivemed/mz_embed_engine.git