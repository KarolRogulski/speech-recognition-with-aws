# Speech Recognition with AWS

## Overview
This machine learning project contains full cycle from project planning through model research to deployment on AWS server. 
The project started with preparing a dataset for training and creating a json file. 
Then the research stage was carried out where the optimization of hyperparameters was performed. 
After finding the best parameters, the model was saved and created with the help of the Flask endpoint,
which, after receiving the .wav file, returns predictions. 
The next step was to deploy the service using uWSGI, which was then placed in a docker container next to the second one with the NGINX engine. 
In the last step, everything was added to the AWS server.

## Development cycle
1. Preparing dataset
2. Model research
3. Creating Flask API
4. Deploying with uWSGI
5. Deploying on Docker with uWSGI
6. Deploying service on Amazon AWS 

### Architecture
![alt text](https://github.com/KarolRogulski/speech-recognition-with-aws/blob/master/res/img/proj-architecture.png?raw=true)

### The words that the model predicts
 * bird
 * bed
 * go
 * five
 * cat
 * down
 * four
 * happy
 * dog
 * eight
 * house
 * left
 * nine
 * marvin
 * off
 * no
 * sheila
 * six
 * on
 * one
 * right
 * seven
 * three
 * stop
 * two
 * tree
 * wow
 * yes
 * up
 * zero
