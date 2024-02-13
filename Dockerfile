FROM tensorflow/tensorflow:1.14.0-gpu-py3
# FROM python:3.6-stretch
# RUN apt-get update && apt-get upgrade -y

MAINTAINER Davood Karimi <davood.karimi@gmail.com>

# install build utilities
# RUN apt-get update 
#RUN apt-get upgrade -y

# set the working directory for containers
WORKDIR  /src/karimi_feta

# Installing python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY icc.py  /src/

# Copy all the files from the project’s root to the working directory
COPY .   /src/
RUN chmod -R 777 /src/

# Running Python Application
CMD ["python3", "/src/karimi_feta_test_aug.py"]

