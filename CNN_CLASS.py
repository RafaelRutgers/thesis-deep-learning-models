#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class InclinationPredictor:
    def __init__(self, input_shape, time_steps, locations, features=128):
        self.input_shape = input_shape
        self.time_steps = time_steps
        self.locations = locations
        self.features = features 
        self.model = None
        self.input_scaler = None

    def custom_loss(self, y_true, y_pred):
    # Main loss (log_cosh for robust regression)
    main_loss = keras.losses.huber(y_true, y_pred, delta=2.0)

    return main_loss

    def build_model(self):
        inputs = layers.Input(shape=self.input_shape, name='input_layer')

        # First Conv block
        x = layers.Conv1D(64, kernel_size=3, padding='same', name='conv1d_1')(inputs)
        x = layers.Activation('tanh', name='activation_tanh_1')(x)
        x = layers.MaxPooling1D(pool_size=2, name='max_pooling1d_1')(x)

        # Second Conv block
        x = layers.Conv1D(128, kernel_size=3, padding='same', name='conv1d_2')(x)
        x = layers.Activation('tanh', name='activation_tanh_2')(x)
        x = layers.MaxPooling1D(pool_size=2, name='max_pooling1d_2')(x)

        # Third Conv block with L2 regularizer
        x = layers.Conv1D(
          256, kernel_size=3, padding='same',
          name='conv1d_3',
          kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = layers.Activation('tanh', name='activation_tanh_3')(x)
        x = layers.MaxPooling1D(pool_size=2, name='max_pooling1d_3')(x)

        # Fourth Conv block
        x = layers.Conv1D(512, kernel_size=3, padding='same', name='conv1d_4')(x)
        x = layers.Activation('tanh', name='activation_tanh_4')(x)
        x = layers.MaxPooling1D(pool_size=2, name='max_pooling1d_4')(x)

        x = layers.Flatten(name='flatten')(x)

        # Dense Block

        x = layers.Dense(512, name='dense1')(x)
        x = layers.Activation('tanh', name='activation_tanh_5')(x)

        x = layers.Dense(self.time_steps * self.locations, name='dense2')(x)

        # Output layer
        outputs = layers.Reshape((self.time_steps, self.locations), name='output_layer')(x)

        model = models.Model(inputs=inputs, outputs=outputs)

        initial_learning_rate = 0.001 # Initial learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

        # Compile the Model with WAPE
        model.compile(
          optimizer='adam',
          loss='huber',
          metrics=['mae', custom_mape(epsilon=1e-3), WAPE()]
        )
        self.model = model


    def scale_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        Scales and fits the input and output training data. Scales the validation and test input and output data.

        Parameters:
        - X: input data of shape (num_samples, 1, 60)
        - y: output data of shape (num_samples, 1800, 50)

        Returns:
        - X_scaled: scaled input data
        - y_scaled: scaled output data
        """

        # Number the samples
        num_samples_train = X_train.shape[0]
        num_samples_val = X_val.shape[0]
        num_samples_test = X_test.shape[0]

        # Flatten input data for scaling
        X_train_flat = X_train.reshape(num_samples_train, -1)
        X_val_flat = X_val.reshape(num_samples_val, -1)
        X_test_flat = X_test.reshape(num_samples_test, -1)

        # Scale and fit training input data
        self.input_scaler = StandardScaler()
        X_train_flat_scaled = self.input_scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_flat_scaled.reshape(num_samples_train, self.input_shape[0], self.input_shape[1])

        # Scale val and test input data
        X_val_flat_scaled = self.input_scaler.transform(X_val_flat)
        X_val_scaled = X_val_flat_scaled.reshape(num_samples_val, self.input_shape[0], self.input_shape[1])

        X_test_flat_scaled = self.input_scaler.transform(X_test_flat)
        X_test_scaled = X_test_flat_scaled.reshape(num_samples_test, self.input_shape[0], self.input_shape[1])

        # Flatten output data for scaling
        y_train_flat = y_train.reshape(num_samples_train, -1)
        y_val_flat = y_val.reshape(num_samples_val, -1)
        y_test_flat = y_test.reshape(num_samples_test, -1)

        # Scale output data
        self.output_scaler = StandardScaler()
        y_train_flat_scaled = self.output_scaler.fit_transform(y_train_flat)
        y_train_scaled = y_train_flat_scaled.reshape(num_samples_train, self.time_steps, self.locations)

        y_val_flat_scaled = self.output_scaler.transform(y_val_flat)
        y_val_scaled = y_val_flat_scaled.reshape(num_samples_val, self.time_steps, self.locations)

        y_test_flat_scaled = self.output_scaler.transform(y_test_flat)
        y_test_scaled = y_test_flat_scaled.reshape(num_samples_test, self.time_steps, self.locations)

        return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled


    def inverse_scale_output(self, y_scaled):
        """
        Inverse transform the scaled output data.

        Parameters:
        - y_scaled: scaled output data

        Returns:
        - y: original output data
        """
        num_samples = y_scaled.shape[0]
        y_flat_scaled = y_scaled.reshape(num_samples, -1)
        y_flat = self.output_scaler.inverse_transform(y_flat_scaled)
        y = y_flat.reshape(num_samples, self.time_steps, self.locations)
        return y

    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=16, epochs=50):
        """
        Train the model.

        Parameters:
        - X_train: training input data
        - y_train: training target data
        - X_val: validation input data (optional)
        - y_val: validation target data (optional)
        - batch_size: batch size for training (default: 16)
        - epochs: number of epochs to train (default: 50)
        """
        reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
        )

        callbacks = [
          tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
          reduce_lr
        ]

        if X_val is not None and y_val is not None:
          self.history = self.model.fit(
              X_train, y_train,
              validation_data=(X_val, y_val),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks
          )
        else:
          self.history = self.model.fit(
              X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks
          )

    def evaluate(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test)
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


    def predict(self, X):
        """
        Make predictions with the model.

        Parameters:
        - X: input data
        """
        return self.model.predict(X)

    def summary(self):
        """
        Print the model summary.
        """
        self.model.summary()

    def save(self, filepath):
        """
        Save the model to a file.

        Parameters:
        - filepath: path to save the model
        """
        self.model.save(filepath)

    def load(self, filepath):
        """
        Load the model from a file.

        Parameters:
        - filepath: path to load the model from
        """
        self.model = models.load_model(filepath)

