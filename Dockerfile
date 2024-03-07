# Use an official Python runtime as a parent image
FROM python:3.9-slim

LABEL Description="ReviveMed Linux environment for mz_embed_engine analysis using Python (R to be added later)"
LABEL tags="revivemed-mz_embed_engine"

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
# COPY . /app
# only copy the requirements_4.txt file
COPY ./requirements_4.txt /app/requirements_4.txt

RUN apt-get update

# install git
RUN apt-get install -y git
# Install useful system packages
RUN apt-get install -y screen htop
RUN apt-get install -y build-essential
RUN apt-get install -y pkg-config
RUN apt-get install -y libmariadb-dev-compat

RUN pip install --upgrade pip
RUN pip install mysqlclient

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements_4.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME World

# Run commands when the container launches
# CMD ["python", "./main.py"]

# Set default command to be jupyter notebook
# CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]

# If you want to make changes to your code without having to rebuild the docker image
# docker run -p 4000:80 -v $(pwd):/app your-image-name

# docker pull dockerrevivemed/mz_embed_engine:v1.1
# docker run -it -p 8080:8080 -v ~/mz_embed_engine:/app dockerrevivemed/mz_embed_engine:v1.1 /bin/bash

# git clone https://jonaheaton@bitbucket.org/revivemed/mz_embed_engine.git
# git stash
# git pull origin dev_jonah

# Add user to the docker group so you don't need sudo and can use vscode to develop:
# https://docs.docker.com/engine/install/linux-postinstall/

