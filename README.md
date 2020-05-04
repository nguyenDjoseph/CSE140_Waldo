# CSE 140 Artificial Intelligence
	 
## Where's Waldo Locator AI
 
 Justin Kelly - Joseph Nguyen - Samuel Gibson

## Overview
The goal of this project is to apply the topics we are learning in class, such as machine learning, feature extraction, and logistic regression to create our own Artificial Intelligence. Specifically, the goal with our project is to create an AI that will solve the Where's Waldo visual puzzles. 

### Additional Information/links 
Project Proposal - https://docs.google.com/document/d/1BC_PnNg2SOmNUZWazsSZt8GOZtdE27H4z9NstkYrr8I/edit

Kaggle Waldo Dataset - https://www.kaggle.com/residentmario/wheres-waldo

OpenCV - https://opencv.org/

Where's Waldo Theory - http://www.randalolson.com/2015/02/03/heres-waldo-computing-the-optimal-search-strategy-for-finding-waldo/

## Models

In this project, we have 2 models in saved_models:

trained_model.hdf5 is our first iteration of the model (the code is located in app.py)
trained_modelv1.hdft is our second iteration (final model) (the code is located in app.py)

To use a different model during the predictor step, update line 17 in predictor/main.py

## Project Usage

To train a model with our dataset, in the main directory run (Using Python 3.7.6):

`python3 app.py`

To test model with our dataset, in the testing directory run: 

`python3 test_models.py`

This will return a 8 64x64 images with the given probability

To use our predictor and detection algorithm, in the predictor directory run:

`python3 main.py {image-name}`

We have 3 images for anyone to test our given model.

Example predictor command:

`python3 main.py test1.jpg`
