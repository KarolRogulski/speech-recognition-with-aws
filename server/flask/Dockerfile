# Pull tensorflow image with Python3
FROM tensorflow/tensorflow:2.1.0-py3
FROM python:3.8

# Set the working directory to /app
WORKDIR /app

# Transfer content from current dir to /app in container
ADD . /app

# Install audio libraries
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

# Install python packages
RUN apt-get update && apt-get install -y apt-transport-https
RUN pip install --upgrade pip
# RUN pip install -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Start uWSGI using config file
CMD ["uwsgi", "app.ini"]