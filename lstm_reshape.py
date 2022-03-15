import numpy
import keras
import pandas as pd

class lstm_reshaper():
    """
    This module provides basic reshaping operation for Keras LSTM input.

    "flatten" takes past values of the same variables as features.
     It takes a 2D dataframe of shape (n_samples, n_features) and a lagging peridod n_lag
     flattens the dataframe by embedding (t-n_lag), (t- (n_lag-1)),... (t-1) into the
     the row. It reshapes the dataframe from a time series structure into a static format.
 
    "transform" takes the result of flatten and reshape according to Kera LSTM input shape.
    It takes a "flattened" 2D dataframe and returns an nd-array in 3D of shape
    ( n_samples - n_lag, n_lag + 1, n_features ) to fit the input size of Keras LSTM model,
    whereas the LSTM layer should take the input_size (n_lag+1,n_features) 
    

    """

    def __init__(self, n_features=1, n_lag=1):
        self.n_features = n_features
        self.n_lag = n_lag

    def flatten(self, data, col_names, n_lag=1, dropnan=True):
        """
        reshape multivariate time series dataframe into an aggregated dataframe with
        past values as features.

        Parameters
        ##########
        :param data: 2D dataframe ( n_samples, n_features )

        :param col_names: list of column names

        :param n_lag: lagging period  T

        :param dropnan: drop NA values. default to True

        :Returns
        ########
        2D  Aggregated dataframe with past values as features
        ( n_samples - n_lag, (n_lag + 1) * n_features)

        """
        df = pd.DataFrame(data)

        self.n_lag = n_lag

        self.n_features = df.shape[1]

        cols, names = list(), list()

        # shifted dataframe put horizontally
        for i in range(n_lag, 0, -1):
            cols.append(df.shift(i))
            names += ['%s(t-%d)' % (col_name, i) for col_name in col_names]
        # concatenate and drop the invalid values due to shift
        cols.append(df)
        names += ['%s(t0)' % col_name for col_name in col_names]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)

        return agg

    def transform(self, X):
        """

        Parameters
        ##########
        2D Dataframe of shape ( n_samples-n_lag, (n_lag+1) * n_features )

        Returns
        #######
        3D Aggregated array-like of shape
        ( n_samples - n_lag, n_lag + 1, n_features )

        """

        n_samples = X.shape[0]

        lstm_X = X.values.reshape(n_samples, self.n_lag + 1, self.n_features)

        return lstm_X


"""
References:
Jason Brownlee "Multivariate Time Series Forecasting with LSTMs in Keras"
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
"""



#example
"""
lr = lstm_reshaper()
df = pd.DataFrame({'A':[5,8,13,20,35,42,65], 'B':[1,4,6,9,14,17,21], 'C':[10,20,30,40,50,60,70]})
print("\n ######################################################")

print("The DataFrame:")
print(df)

print("\n ######################################################")

flat = lr.flatten(df,df.columns,n_lag=2)
print("Flattened Dataframe with n_lag=2")
print(flat.head())

print("\n ######################################################")

lstm_input = lr.transform(flat)
print("LSTM layer input")
print(lstm_input)
"""
