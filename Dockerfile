# Use an official Python runtime as a parent image
FROM python:3.9

# Update package list
RUN apt-get update

# Install necessary system packages
RUN apt-get install -y screen htop

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

############################################################
###### Python Package Installation ######

# Install python and pip
RUN apt-get update && apt-get install -y python3 python3-pip

RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libudunits2-dev
# Install pandoc
RUN apt-get update && apt-get install -y pandoc
# Install virtualenv
RUN pip3 install virtualenv

# Create a virtual environment and activate it
RUN python3 -m virtualenv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Now install your Python packages(~1 min)
RUN pip install -r requirements_3.txt


############################################################
###### R Installation ######
RUN apt-get update && apt-get upgrade -y
# Install R (~1 min)
RUN apt-get install -y r-base

### Option 1: Install R packages using a requirements file
# Install R packages
# Copy the r-requirements.txt file into the Docker image
# COPY r-requirements.txt /app/r-requirements.txt
# RUN while read requirement; do Rscript -e "install.packages('$requirement', repos='http://cran.rstudio.com/')"; done < /app/r-requirements.txt

### Option 2: Install R packages using a script (~ 30 min)
# Follow instructions from https://github.com/hhabra/metabCombiner
# Copy the install_script.R file
COPY install_script.R /app/install_script.R

# Run the install_script.R file (> 15 min?!)
RUN Rscript /app/install_script.R

############################################################
###### Set the Default Commands and expose ports ######

# Make port 80 available to the world outside this container
EXPOSE 80

# set the default command to execute app.py when the container launches
# CMD ["python", "app.py"]

# Set default command to be jupyter notebook
# CMD ["jupyter", "notebook", "--ip='*'", "--port=8888", "--no-browser", "--allow-root"]

############################################################
###### NOTES ######

# Assuming default command is jupyter notebook
# To run the Docker container, use the following command:
# docker run -p 8888:8888 -v $(pwd):/app my-python-env

# This will start a Jupyter Notebook server that you can access at localhost:8888 in your browser. 
# The -v $(pwd):/app option mounts your current directory into the /app directory in the container, 
# so you can work on files in your current directory.


# Once the image is built, you can run a container from it using the docker run command. 
# For example, to run a container from the my_image image, you would use:
# >> docker run -it --rm my_image


####
#If I want to save changes made in a docker container (such as installing new python modules), the fastest way is to Commit the changes to a new Docker image: 
# After installing the new Python modules, you can commit the changes to a new Docker image using the docker commit command.
# This will create a new Docker image that includes the new Python modules. You can then use this new image to create new containers.