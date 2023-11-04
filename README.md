# Environment requirements

The code needs to be built on Python to run, including some  necessary packages （pytorch 1.2 version (python 3.7), pandas, matplotlib, numpy, and scikit-learn).

# File introduction

demo_data - constructing demo dataset

data_loader_channels.py - loading the produced dataset

CNN_constraint.py - constructing CNN models with geological constraints

CNN_constraint_mian.py - training the constructed model with geological constraints

utils - eval.py - calculating accuracy

utils - misc.py - auxiliary function

# Reproduction Guide

Our experimental data includes confidential geological data and confidentiality agreements have been signed with the relevant organizations. Although we are unable to provide the original data, we have produced demo data in the exact same format as the experimental data to help researchers reproduce work in regional survey.

The demo data includes training data (train_dataset_extend) and validation data (valid_dataset_extend). The training data is saved in the train_dataset_extend folder and includes the same number and size of positive and negative samples saved in subfolders 1 and 0, respectively. Positive samples were sampled in areas with favorable mineralization, while negative samples were sampled in areas with low mineralization probability. Each npz file consists of four parts of: data, term, term1, and term2 (lines27-30 in data_loader_channels.py). Data is the training data with the size of 7×7×42, where 7 is the window size and 42 is the number of channels. Term, term1, and term2 are geologic prior knowledge representations embedded feature extractor, classifier, and loss function of model. The arrangement of validation data is the same as the training data. The format of the demo data is the same as the experimental data format, if researchers want to reproduce the code, you can construct training samples according to the demo data and place them in the corresponding folder. After the dataset is made, the model structure is built using CNN_constraint.py, and then CNN_constraint_main.py is run to train the model.

# Reference

The code is partly from the paper "Applications of data augmentation in mineral prospectivity prediction based on convolutional neural networks" .
