
#1. Install Docker (if you don’t have it yet)
# # Update packages
# sudo apt update

# # Install Docker
# sudo apt install docker.io

# # Start Docker service
# sudo systemctl start docker

# # Enable Docker to run on boot
# sudo systemctl enable docker

docker --version

# 2. Create a Dockerfile
cd /path/to/your/project
touch Dockerfile


# 3. Write the Dockerfile
# Start with a base image
FROM ubuntu:20.04

# Set the maintainer of the image
LABEL maintainer="your-email@example.com"

# Update the package manager and install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    vim

# Copy all files from the current directory into the container
COPY . /app

# Set the working directory
WORKDIR /app

# Install Python dependencies (if you have a requirements.txt)
RUN pip3 install -r requirements.txt

# Expose a port if needed (optional)
EXPOSE 5000

# Command to run when the container starts
CMD ["python3", "your_script.py"]



# 4. Build the Docker Image
# Navigate to the directory with your Dockerfile
cd /path/to/your/project

# Build the Docker image
docker build -t your_dockerhub_username/your_image_name .


# 5. Log into Docker Hub
docker login

# 6. Tage the image
docker tag your_dockerhub_username/your_image_name your_dockerhub_username/your_image_name:v1.0


# 7. Push the Image to Docker Hub
docker push your_dockerhub_username/your_image_name

docker push your_dockerhub_username/your_image_name:v1.0

# 8. Run the Docker Container
docker pull your_dockerhub_username/your_image_name:v1.0

docker run -it your_dockerhub_username/your_image_name:v1.0

# 9. Stop the Docker Container
# List all running containers
docker ps


# Managing Dependencies: Ensure that your requirements.txt or other dependency management files are in your project directory if you're working with Python.

# Testing Locally: Before pushing the image to Docker Hub, you can run and test it locally using docker run.
