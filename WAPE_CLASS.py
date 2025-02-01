#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class WAPE(tf.keras.metrics.Metric):
    def __init__(self, name='wape', **kwargs):
        super(WAPE, self).__init__(name=name, **kwargs)
        # Initialize variables to store the sum of absolute errors and sum of absolute true values
        self.total_error = self.add_weight(name='total_error', initializer='zeros')
        self.total_true = self.add_weight(name='total_true', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Compute absolute errors and absolute true values
        error = tf.reduce_sum(tf.abs(y_true - y_pred))
        true_sum = tf.reduce_sum(tf.abs(y_true))

        # Update the state variables
        self.total_error.assign_add(error)
        self.total_true.assign_add(true_sum)

    def result(self):
        # Compute WAPE: (Total Absolute Error / Total Absolute True) * 100%
        return (self.total_error / (self.total_true + 1e-8)) * 100.0

    def reset_states(self):
        # Reset the state variables at the start of each epoch
        self.total_error.assign(0.0)
        self.total_true.assign(0.0)

