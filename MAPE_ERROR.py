#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class MAPE(tf.keras.metrics.Metric):
    def __init__(self, name='mape', epsilon=1e-6, **kwargs):
        super(MAPE, self).__init__(name=name, **kwargs)
        self.epsilon = epsilon  # Custom epsilon to avoid division by zero
        
        # Initialize variables to store sum of absolute percentage errors and count
        self.total_error = self.add_weight(name='total_error', initializer='zeros')
        self.total_count = self.add_weight(name='total_count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute absolute percentage error (APE)
        diff = tf.abs(y_true - y_pred)
        denom = tf.clip_by_value(tf.abs(y_true), self.epsilon, tf.float32.max)  # Avoid division by zero
        ape = (diff / denom) * 100.0  # Compute percentage error

        # Sum APE over batch
        sum_ape = tf.reduce_sum(ape)
        count = tf.cast(tf.size(y_true), tf.float32)

        # Update state variables
        self.total_error.assign_add(sum_ape)
        self.total_count.assign_add(count)

    def result(self):
        # Compute MAPE: (Sum of APE / Count)
        return self.total_error / (self.total_count + 1e-8)

    def reset_states(self):
        # Reset the state variables at the start of each epoch
        self.total_error.assign(0.0)
        self.total_count.assign(0.0)

