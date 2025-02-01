#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class PlatformTiltPredictor:
    def __init__(self, num_features=300, batch_size=8, epochs=20, time_steps=3600, window_size=300, step_size=300):
        """
        Initializes the predictor with hyperparameters.

        Parameters:
        - window_size (int): Number of time steps in each window.
        - step_size (int): Step size for the sliding window.
        - batch_size (int): Batch size for training.
        - epochs (int): Number of epochs for training.
        """
        self.num_features = num_features
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.wave_height_scaler = StandardScaler()
        self.model = None
        self.time_steps = time_steps
        self.lstm_features = 256
        self.window_size = window_size
        self.step_size = step_size


    def preprocess_data(self, X_raw, y_raw, wh_raw, season_raw):
        """
        Generates sliding window sequences from the raw data.

        Parameters:
        - X_raw (np.ndarray): Raw input data of shape (num_samples, total_time_steps, num_input_features).
        - y_raw (np.ndarray): Raw output data of shape (num_samples, total_time_steps, num_output_features).

        Returns:
        - X_windows (np.ndarray): Input sequences for the model.
        - y_windows (np.ndarray): Output sequences for the model.
        """
        num_samples, total_time_steps, num_input_features = X_raw.shape
        _, _, num_output_features = y_raw.shape

        # Generate sliding windows
        X_windows = []
        y_windows = []
        wh_windows = []

        for sample_idx in range(num_samples):
            for start in range(0, total_time_steps - self.window_size + 1, self.step_size):
                end = start + self.window_size
                X_windows.append(X_raw[sample_idx, start:end, :])
                y_windows.append(y_raw[sample_idx, start:end, :])
                wh_windows.append(wh_raw[sample_idx, start:end, :])

        X_windows = np.array(X_windows)
        y_windows = np.array(y_windows)
        wh_windows = np.array(wh_windows)

        season_windows = []
        for sample_idx in range(num_samples):
            for start in range(0, total_time_steps - self.window_size + 1, self.step_size):
                season_windows.append(season_raw[sample_idx])

        season_windows = np.array(season_windows)

        return X_windows, y_windows, wh_windows, season_windows

    def scale_data(self, X_train, y_train, wh_train,
                  X_val=None, y_val=None, wh_val=None,
                  X_test=None, y_test=None, wh_test=None):
        """
        Scales the data using fit_transform on training data
        or transform on validation/test data if training data is None.
        """

        # Prepare return variables
        X_train_scaled, y_train_scaled, wh_train_scaled = None, None, None
        X_val_scaled, y_val_scaled, wh_val_scaled = None, None, None
        X_test_scaled, y_test_scaled, wh_test_scaled = None, None, None

        # ------------------------------------------
        # 1) If X_train is not None, fit the scalers
        # ------------------------------------------
        if X_train is not None and y_train is not None and wh_train is not None:
            # Flatten input data for scaling
            num_samples_train = X_train.shape[0]
            X_train_flat = X_train.reshape(-1, X_train.shape[2])
            X_train_flat_scaled = self.input_scaler.fit_transform(X_train_flat)
            X_train_scaled = X_train_flat_scaled.reshape(X_train.shape)

            # Flatten output data for scaling
            y_train_flat = y_train.reshape(-1, y_train.shape[2])
            y_train_flat_scaled = self.output_scaler.fit_transform(y_train_flat)
            y_train_scaled = y_train_flat_scaled.reshape(y_train.shape)

            # Flatten wave height data for scaling
            wh_train_flat = wh_train.reshape(-1, wh_train.shape[2])
            wh_train_flat_scaled = self.wave_height_scaler.fit_transform(wh_train_flat)
            wh_train_scaled = wh_train_flat_scaled.reshape(wh_train.shape)
        else:
            # If we have no training data, we assume the scalers are already fitted
            # (e.g., from a previous training phase). So we skip the .fit
            pass

        # ------------------------------------------
        # 2) Scale X_val, y_val, wh_val with .transform
        # ------------------------------------------
        if X_val is not None:
            X_val_flat = X_val.reshape(-1, X_val.shape[2])
            X_val_flat_scaled = self.input_scaler.transform(X_val_flat)
            X_val_scaled = X_val_flat_scaled.reshape(X_val.shape)

        if y_val is not None:
            y_val_flat = y_val.reshape(-1, y_val.shape[2])
            y_val_flat_scaled = self.output_scaler.transform(y_val_flat)
            y_val_scaled = y_val_flat_scaled.reshape(y_val.shape)

        if wh_val is not None:
            wh_val_flat = wh_val.reshape(-1, wh_val.shape[2])
            wh_val_flat_scaled = self.wave_height_scaler.transform(wh_val_flat)
            wh_val_scaled = wh_val_flat_scaled.reshape(wh_val.shape)

        # ------------------------------------------
        # 3) Scale X_test, y_test, wh_test with .transform
        # ------------------------------------------
        if X_test is not None:
            X_test_flat = X_test.reshape(-1, X_test.shape[2])
            X_test_flat_scaled = self.input_scaler.transform(X_test_flat)
            X_test_scaled = X_test_flat_scaled.reshape(X_test.shape)

        if y_test is not None:
            y_test_flat = y_test.reshape(-1, y_test.shape[2])
            y_test_flat_scaled = self.output_scaler.transform(y_test_flat)
            y_test_scaled = y_test_flat_scaled.reshape(y_test.shape)

        if wh_test is not None:
            wh_test_flat = wh_test.reshape(-1, wh_test.shape[2])
            wh_test_flat_scaled = self.wave_height_scaler.transform(wh_test_flat)
            wh_test_scaled = wh_test_flat_scaled.reshape(wh_test.shape)

        return (
            X_train_scaled, y_train_scaled, wh_train_scaled,
            X_val_scaled, y_val_scaled, wh_val_scaled,
            X_test_scaled, y_test_scaled, wh_test_scaled
        )
    def build_model(self, num_wave_height_features=2, num_output_features=50):
        """
        Builds the LSTM model.

        Parameters:
        - num_wave_height_features (int): Number of wave height features.
        - num_output_features (int): Number of output features per time step.
        """
        # Main Spectral Density Input
        spectral_input = layers.Input(shape=(self.window_size, 1), name='spectral_input')

        # First Conv block
        x = layers.Conv1D(64, kernel_size=3, padding='same', name='conv1d_1')(spectral_input)
        x = layers.Activation('tanh', name='activation_tanh_1')(x)
        x = layers.MaxPooling1D(pool_size=2, name='max_pooling1d_1')(x)

        # Second Conv block
        x = layers.Conv1D(128, kernel_size=3, padding='same', name='conv1d_2')(x)
        x = layers.Activation('tanh', name='activation_tanh_2')(x)
        x = layers.MaxPooling1D(pool_size=2, name='max_pooling1d_2')(x)

        # Third Conv block
        x = layers.Conv1D(256, kernel_size=3, padding='same', name='conv1d_3')(x)
        x = layers.Activation('tanh', name='activation_tanh_3')(x)
        x = layers.MaxPooling1D(pool_size=2, name='max_pooling1d_3')(x)

        # Fourth Conv block
        x = layers.Conv1D(512, kernel_size=3, padding='same', name='conv1d_4')(x)
        x = layers.Activation('tanh', name='activation_tanh_4')(x)
        x = layers.MaxPooling1D(pool_size=2, name='max_pooling1d_4')(x)

        # Flatten and Dense Layers
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(512, name='dense1')(x)
        x = layers.Activation('tanh', name='activation_tanh_5')(x)
        x = layers.Dense(self.window_size * self.lstm_features, name='dense2')(x)
        x = layers.Reshape((self.window_size, self.lstm_features), name='reshape')(x)

        # Wave Height Input
        wave_height_input = layers.Input(shape=(self.window_size, num_wave_height_features), name='wave_height_input')

        # Concatenate Spectral Features and Wave Height
        x_combined = layers.Concatenate(axis=-1, name='concatenate')([x, wave_height_input])

        # First LSTM Block
        x = layers.LSTM(256, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name='lstm_1')(x_combined)
        x = layers.BatchNormalization(name='batch_norm_1')(x)

        # Second LSTM Block
        x = layers.LSTM(512, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', name='lstm_2')(x)

        # Output Layer
        outputs = layers.TimeDistributed(layers.Dense(num_output_features), name='output_layer')(x)

        # Define the Model
        model = models.Model(inputs=[spectral_input, wave_height_input], outputs=outputs, name='PlatformTiltPredictor_Model')

        # Compile the Model with WAPE
        model.compile(
            optimizer='adam',
            loss='huber',
            metrics=['mae', custom_mape(epsilon=1e-3), WAPE()]
        )
        self.model = model


    def train(self, X_train, wh_train, y_train, X_val=None, wh_val=None, y_val=None):
        """
        Trains the model.

        Parameters:
        - X_train (np.ndarray): Training input data.
        - y_train (np.ndarray): Training output data.
        - X_val (np.ndarray): Validation input data.
        - y_val (np.ndarray): Validation output data.
        """


        # Set up callbacks, including learning rate scheduler
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=1,
                min_lr=1e-6,
                mode='min'
            )
        ]

        if X_val is not None and y_val is not None:
          self.history = self.model.fit(
              [X_train, wh_train], y_train,
              validation_data=([X_val, wh_val], y_val),
              batch_size=self.batch_size,
              epochs=self.epochs,
              callbacks=callbacks
          )
        else:
          self.history = self.model.fit(
              [X_train, wh_train], y_train,
              batch_size=self.batch_size,
              epochs=self.epochs,
              callbacks=callbacks
          )

    def evaluate(self, X_test, y_test, wh_test):
        results = self.model.evaluate([X_test, wh_test], y_test, batch_size=self.batch_size)
        # If your metrics are [loss, mae, mape, wape], then:
        test_loss, test_mae, test_mape, test_wape = results

        print(f'Test Loss: {test_loss:.6f}')
        print(f'Test MAE: {test_mae:.4f}')
        print(f'Test MAPE: {test_mape:.4f}%')
        print(f'Test WAPE: {test_wape:.4f}%')

        return {
            'Test Loss': test_loss,
            'Test MAE': test_mae,
            'Test MAPE': test_mape,
            'Test WAPE': test_wape
        }

    def predict(self, X_input, wh_input):
        """
        Makes predictions using the trained model.

        Parameters:
        - X_input (np.ndarray): Input data.
        - wh_input (np.ndarray): Wave height data.

        Returns:
        - y_pred (np.ndarray): Predicted output data in original scale.
        """

        # Make predictions
        y_pred_scaled = self.model.predict([X_input, wh_input], batch_size=self.batch_size)

        # Reshape and inverse scale the predictions
        y_pred_flat = y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1])
        y_pred_flat_orig = self.output_scaler.inverse_transform(y_pred_flat)
        y_pred = y_pred_flat_orig.reshape(y_pred_scaled.shape)

        return y_pred

    def summary(self):
      """
      Print the model summary.
      """
      self.model.summary()

    def save_model(self, filepath):
        """
        Saves the trained model to a file.

        Parameters:
        - filepath (str): Path to save the model.
        """
        self.model.save(filepath)

    def load_model(self, filepath):
        """
        Loads a trained model from a file.

        Parameters:
        - filepath (str): Path to the saved model.
        """
        self.model = tf.keras.models.load_model(filepath)

