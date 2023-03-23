# lstm_reshape
### This module provides a single function that converts a multivariate time series dataframe into a supervised learning - style dataframe by flattening and shifting each series along a given lagging period. For example, when one tries to use keras on a 2-D tensor of shape (total_time_steps, no_features), it's required to reshape the tensor into a 3-D tensor of input_shape (no_features, time_steps, batch_size).  