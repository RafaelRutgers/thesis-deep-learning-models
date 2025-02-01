# thesis-deep-learning-models
Codes for the CNN, LSTM, and CNN-LSTM hybrid models for the prediction of the tilt response of Very Large Floating Structures. All models predict the tilt response at 50 locations along a fictive OFPV platform for each second during a 1 hour storm. Only the sea state is used as a variable input for all the models. These are spectral density values for the CNN and CNN-LSTM hybrid. Two wave elevation values per second are used as a variable input for the LSTM and CNN-LSTM hybrid. The latter thus includes two types of inputs.

The required packages to run these codes are given in the file Packages.py. The codes are developed in Google Colab. When not using Google Colab, the concerned import packages can be ignored.

The data used, is created by a FE-FSI model, and can be sent upon request. Make sure to properly prepare the data: split and scale the data, and stratify if wanted.

The LSTM_CLASS.py and CNN-LSTM-HYBRID_CLASS.py include a windowing function inside the class. Make sure to call this function before scaling and training. In the CNN-LSTM hybrid class the spectral density values are concatenated as many times to match the time dimension of one storm (3600 seconds).
