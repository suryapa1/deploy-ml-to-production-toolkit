# The base image to build the Docker container from
FROM jupyter/datascience-notebook:177037d09156
MAINTAINER Armen Donigian (donigian@gmail.com)

# launchbot-specific labels
LABEL name.launchbot.io="Deploy ML to Production Toolkit"
LABEL workdir.launchbot.io="/home/jovyan"
LABEL description.launchbot.io="Deploy ML to Production Toolkit"
LABEL 8888.port.launchbot.io="Start Tutorial"

COPY . /home/jovyan
USER root
RUN apt-get update && apt-get install -y \
    gcc \
    libx11-dev  \ 
    git-core \
    apt-utils \
    libboost-program-options-dev \ 
    zlib1g-dev \
    g++ 
RUN chmod -R 777 /home/jovyan

USER jovyan
RUN conda env create -f /home/jovyan/env.yml

#Set the working directory
WORKDIR /home/jovyan/

# Add files
COPY solutions /home/jovyan/solutions

# Allow user to write to directory
USER root
RUN chown -R $NB_USER /home/jovyan \
    && chmod -R 774 /home/jovyan
USER $NB_USER

RUN pip install -r requirements.txt
# Expose the notebook port
EXPOSE 8888

#RUN git clone git://github.com/JohnLangford/vowpal_wabbit.git && cd vowpal_wabbit && make && make install
# Start the notebook server
CMD jupyter notebook --no-browser --port 8888 --ip=* --NotebookApp.token='' --NotebookApp.disable_check_xsrf=True --NotebookApp.iopub_data_rate_limit=1.0e10


