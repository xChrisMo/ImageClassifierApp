# Flower Classifier with Deep Learning using TensorFlow

This project aims to simulate real world deep learning problems of minimal data being expected to generalise well. 

## Data

Using the Oxford 102 Flowers dataset, `oxford_flowers102`, directly from TensorFlow.
The main context around this exact dataset is that the data used to train the model, `training_set`, is less than one-fifth of what the trained model is tested on. Prioritising generalisation and sublte data augmentation.

## Project Files

       |-- test_images
       |-- Project_image.ipynb
       |-- final_model.h5
       |-- label_map.json
       |-- predict.py
       |-- utility.py

### File Description

* test_images: directory containing random images on which model was evaluated
* Project_image.ipynb: full Jupyter notebook for all augmentating of data, training, evaluating and packaging of model
* final_model.h5: Trained model in .h5 file format
* label_map.json: JSON file for mapping flowers and their classes
* predict.py; .py file for testing model in Terminal
* utility.py: Helper file for functions

## How to run
Having all dependencies installed, imply fork the .ipynb notebook and run in real time to replicate analysis


## Licensing
This project is for personal and educational purposes, feel free to use this repo anyhow
