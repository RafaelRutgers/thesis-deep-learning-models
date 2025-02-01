#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class PlatformTiltPredictor:
    def __init__(self, window_size=20, step_size=20, batch_size=8, epochs=20):
        self.window_size = window_size
        self.step_size = step_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.model = None

    def preprocess_data(self, X_raw, y_raw, season_raw):
        num_samples, total_time_steps, num_input_features = X_raw.shape
        _, _, num_output_features = y_raw.shape

        X_windows = []
        y_windows = []
        for sample_idx in range(num_samples):
            for start in range(0, total_time_steps - self.window_size + 1, self.step_size):
                end = start + self.window_size
                X_windows.append(X_raw[sample_idx, start:end, :])
                y_windows.append(y_raw[sample_idx, start:end, :])

        X_windows = np.array(X_windows)
        y_windows = np.array(y_windows)

        season_windows = []
        for sample_idx in range(num_samples):
            for start in range(0, total_time_steps - self.window_size + 1, self.step_size):
                season_windows.append(season_raw[sample_idx])

        season_windows = np.array(season_windows)

        return X_windows, y_windows, season_windows

    def scale_data(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        num_samples_train = X_train.shape[0]
        X_train_flat = X_train.reshape(-1, X_train.shape[2])
        self.input_scaler = StandardScaler()
        X_train_flat_scaled = self.input_scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_flat_scaled.reshape(X_train.shape)

        y_train_flat = y_train.reshape(-1, y_train.shape[2])
        self.output_scaler = StandardScaler()
        y_train_flat_scaled = self.output_scaler.fit_transform(y_train_flat)
        y_train_scaled = y_train_flat_scaled.reshape(y_train.shape)

        if X_val is not None and y_val is not None:
            X_val_flat = X_val.reshape(-1, X_val.shape[2])
            X_val_flat_scaled = self.input_scaler.transform(X_val_flat)
            X_val_scaled = X_val_flat_scaled.reshape(X_val.shape)

            y_val_flat = y_val.reshape(-1, y_val.shape[2])
            y_val_flat_scaled = self.output_scaler.transform(y_val_flat)
            y_val_scaled = y_val_flat_scaled.reshape(y_val.shape)
        else:
            X_val_scaled = None
            y_val_scaled = None

        if X_test is not None and y_test is not None:
            X_test_flat = X_test.reshape(-1, X_test.shape[2])
            X_test_flat_scaled = self.input_scaler.transform(X_test_flat)
            X_test_scaled = X_test_flat_scaled.reshape(X_test.shape)

            y_test_flat = y_test.reshape(-1, y_test.shape[2])
            y_test_flat_scaled = self.output_scaler.transform(y_test_flat)
            y_test_scaled = y_test_flat_scaled.reshape(y_test.shape)
        else:
            X_test_scaled = None
            y_test_scaled = None

        return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled

    def custom_loss(self, y_true, y_pred):
        main_loss = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
        return main_loss

    def build_model(self, num_input_features=2, num_output_features=50):
        inputs = layers.Input(shape=(self.window_size, num_input_features), name='input_layer')
        x = layers.LSTM(64, return_sequences=True, name='lstm_1')(inputs)
        x = layers.LSTM(128, return_sequences=True, name='lstm_2')(x)
        x = layers.BatchNormalization(name='batch_norm_1')(x)
        x = layers.LSTM(256, return_sequences=True, name='lstm_3')(x)
        x = layers.BatchNormalization(name='batch_norm_2')(x)
        x = layers.LSTM(512, return_sequences=True, name='lstm_4')(x)
        x = layers.TimeDistributed(layers.Dense(512), name='time_distributed_dense_1')(x)
        outputs = layers.TimeDistributed(layers.Dense(num_output_features), name='time_distributed_dense_2')(x)

        model = Model(inputs=inputs, outputs=outputs, name="LSTM_Model")
        model.compile(
            optimizer='adam',
            loss='huber',
            metrics=['mae', custom_mape(epsilon=1e-3), WAPE()]  # Added WAPE here
        )
        self.model = model

    def train(self, X_train, y_train, X_val=None, y_val=None):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=1,
                min_lr=1e-6
            )
        ]

        if X_val is not None and y_val is not None:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=callbacks
            )
        else:
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=callbacks
            )

    def evaluate(self, X_test, y_test):
        results = self.model.evaluate(X_test, y_test, batch_size=self.batch_size)
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

    def predict_full_sequence(self, X_input, total_time_steps):
        y_pred_scaled = self.model.predict(X_input)
        num_windows, window_size, num_output_features = y_pred_scaled.shape

        y_pred_flat = y_pred_scaled.reshape(-1, num_output_features)
        y_pred_flat_orig = self.output_scaler.inverse_transform(y_pred_flat)
        y_pred = y_pred_flat_orig.reshape(num_windows, window_size, num_output_features)

        num_samples = X_input.shape[0] // ((total_time_steps - self.window_size) // self.step_size + 1)
        y_full_pred_sum = np.zeros((num_samples, total_time_steps, num_output_features))
        counts = np.zeros((num_samples, total_time_steps, 1))

        for idx in range(num_windows):
            sample_idx = idx // ((total_time_steps - self.window_size) // self.step_size + 1)
            window_start = (idx % ((total_time_steps - self.window_size) // self.step_size + 1)) * self.step_size
            window_end = window_start + self.window_size

            y_full_pred_sum[sample_idx, window_start:window_end, :] += y_pred[idx]
            counts[sample_idx, window_start:window_end, :] += 1

        counts[counts == 0] = 1
        y_full_pred = y_full_pred_sum / counts
        return y_full_pred

    def summary(self):
        self.model.summary()

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = tf.keras.models.load_model(filepath)

