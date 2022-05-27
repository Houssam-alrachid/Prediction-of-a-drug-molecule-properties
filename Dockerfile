# Mention the base image 
FROM continuumio/anaconda3:4.4.0

LABEL Author, Houssam ALRACHID

ENV APP_HOME /servier

# Copy the current folder structure and content to docker folder
COPY . $APP_HOME

# Expose the port within docker 
EXPOSE 5000

# Set current working directory
WORKDIR $APP_HOME

# Install the required libraries
RUN pip install -r requirements.txt &&\
    conda install -c conda-forge rdkit

# Container start up command
CMD python flask_api.py