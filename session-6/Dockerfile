# set base image (host OS)
FROM python:3.8-slim

# set the working directory in the container
WORKDIR /tmp

# copy the dependencies file to the working directory
COPY requirements.txt .
COPY data/housing.csv /tmp/data/housing.csv

# install dependencies
RUN pip install -r requirements.txt

# Copy code to the working directory
COPY src/ /tmp/src

# command to run on container start
ENTRYPOINT ["python", "/tmp/src/entrypoint.py"]
