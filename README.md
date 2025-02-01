# thesis-deep-learning-models
This repository contains the code for CNN, LSTM, and CNN-LSTM hybrid models used to predict the tilt response of Very Large Floating Structures (VLFSs).

Overview

All models predict the tilt response at 50 locations along a fictive Offshore Floating Photovoltaic (OFPV) platform for every second during a one-hour storm. The only variable input for all models is the sea state:
- The CNN and CNN-LSTM hybrid models use spectral density values as input.
- The LSTM and CNN-LSTM hybrid models use two wave elevation values per second as input.
- The CNN-LSTM hybrid combines both spectral density values and wave elevation data, incorporating two types of inputs.

Requirements & Installation
- The required packages are listed in requirements.txt.
- The modules used in this project are provided in Modules.py.
- The code was developed using Google Colab.
- If running the code outside Google Colab, you can safely ignore Colab-specific imports.

Data Availability & Preparation

The dataset used in this study was generated using a Finite Element - Fluid-Structure Interaction (FE-FSI) model and is available upon request.

If using the dataset, ensure proper preprocessing, including:
- Splitting and scaling the data.
- Stratifying, if necessary.

Model-Specific Notes
- The LSTM_CLASS.py and CNN-LSTM-HYBRID_CLASS.py files contain an internal windowing function.
- Ensure this function is called before scaling and training the models.
- In the CNN-LSTM hybrid class, spectral density values are repeated to match the time dimension of a full storm (3600 seconds).
