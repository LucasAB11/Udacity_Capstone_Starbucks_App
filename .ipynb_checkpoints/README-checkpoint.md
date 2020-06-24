# Udacity_Capstone_Starbucks_App
The goal of this project is to use datasets from the Starbucks Rewards App to make predictions about which offers should be sent to which customers. The datasets provide information about the available offers, the customers and a transcript over a one month period with all offers made to customers and transactions made by the customers. A Linear Learner, an XGB Boost model and a pytorch neural network are used to make predictions on how a customer is likely to react to a offer based on the customer and offer characteristics. The customer reactions fall into four categories based on whether the customer viewed and whether the customer completed the offer. The prediction of the customer's reaction to an offer is then used to make a recommendation about which offer should be sent to which customer.


Python Version: 3.8

Used Libraries:
numpy
pandas
matplotlib.pyplot
seaborn
math
json
os
sklearn.preprocessing
sklearn.model_selection
sklearn.metrics
sagemaker
sagemaker.amazon.amazon_estimator
sagemaker.pytorch
torch

