# lstm_reshaper
### This module provides a single function that converts a multivariate time series dataframe into a supervised learning - style dataframe 
### When using Keras LSTM for sequence modeling tasks, the input data is typically represented as a 2D tensor with shape (num_samples, num_timesteps). Here, num_samples represents the number of data points in the dataset, and num_timesteps represents the length of each sequence.

### However, LSTM requires a 3D input shape of (num_samples, num_timesteps, num_features), where num_features represents the number of features in each time step. To convert the 2D tensor to a 3D tensor, we can use this lstm reshaper
