# import libraries
import os
import numpy as np
import torch
from six import BytesIO

# import model from model.py, by name
from model import MultiClassClassifier

# default content type is numpy array
NP_CONTENT_TYPE = 'application/x-npy'


# Provided model load function
def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiClassClassifier(model_info['input_features'], model_info['output_dim'])

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Prep for testing
    model.to(device).eval()

    print("Done loading model.")
    return model


# Provided predict function
def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process input_data so that it is ready to be sent to our model.
    data = input_data.float()
    data = data.to(device)

    # Put the model into evaluation mode
    model.eval()

    # Compute the result of applying the model to the input data
    # The neural network has 4 output nodes. To calculate the probabilities of the labels, a softmax has to be applied
    out = model(data)
    out_pred_softmax = torch.log_softmax(data, dim = 1)
    #once the softmax is applied, pick the max and return that label
    _, out_pred_tags = torch.max(y_pred_softmax, dim = 1)

    return out_pred_tags.numpy().squeeze()