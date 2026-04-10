"""Minimal network modules used by MissBGM."""

from __future__ import annotations

import tensorflow as tf


class BaseFullyConnectedNet(tf.keras.Model):
    """Basic MLP used for encoders and the missingness network."""

    def __init__(self, input_dim, output_dim, model_name, nb_units=None, batchnorm=False):
        super(BaseFullyConnectedNet, self).__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.model_name = model_name
        self.nb_units = list(nb_units or [256, 256, 256])
        self.batchnorm = bool(batchnorm)
        self.hidden_layers = []

        for units in self.nb_units:
            dense = tf.keras.layers.Dense(
                units=units,
                activation=None,
                kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                bias_regularizer=tf.keras.regularizers.L2(1e-4),
            )
            norm = tf.keras.layers.BatchNormalization()
            self.hidden_layers.append((dense, norm))
        self.output_layer = tf.keras.layers.Dense(
            units=self.output_dim,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            bias_regularizer=tf.keras.regularizers.L2(1e-4),
        )
        self(tf.zeros((1, self.input_dim), dtype=tf.float32))

    def call(self, inputs, training=True):
        x = tf.cast(inputs, tf.float32)
        for dense, norm in self.hidden_layers:
            x = dense(x)
            if self.batchnorm:
                x = norm(x, training=training)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return self.output_layer(x)


class BaseVariationalNet(tf.keras.Model):
    """Diagonal Gaussian network returning mean and variance."""

    def __init__(self, input_dim, output_dim, model_name, nb_units=None):
        super(BaseVariationalNet, self).__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.model_name = model_name
        self.nb_units = list(nb_units or [256, 256, 256])
        self.hidden_layers = [tf.keras.layers.Dense(units=units, activation=None) for units in self.nb_units]
        self.norm_layer = tf.keras.layers.BatchNormalization()
        self.mean_layer = tf.keras.layers.Dense(units=self.output_dim, activation=None)
        self.var_layer = tf.keras.layers.Dense(units=self.output_dim, activation=None)

    def call(self, inputs, eps=1e-6, training=True):
        x = self.norm_layer(tf.cast(inputs, tf.float32), training=training)
        for dense in self.hidden_layers:
            x = dense(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        mean = self.mean_layer(x)
        var = tf.nn.softplus(self.var_layer(x)) + eps
        return mean, var

    def reparameterize(self, mean, var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.sqrt(var) + mean


class Discriminator(tf.keras.Model):
    """Simple fully connected discriminator."""

    def __init__(self, input_dim, model_name, nb_units=None, batchnorm=True):
        super(Discriminator, self).__init__()
        self.input_dim = int(input_dim)
        self.model_name = model_name
        self.nb_units = list(nb_units or [256, 256])
        self.batchnorm = bool(batchnorm)
        self.hidden_layers = []

        for units in self.nb_units:
            dense = tf.keras.layers.Dense(units=units, activation=None)
            norm = tf.keras.layers.BatchNormalization()
            self.hidden_layers.append((dense, norm))
        self.output_layer = tf.keras.layers.Dense(units=1, activation=None)
        self(tf.zeros((1, self.input_dim), dtype=tf.float32))

    def call(self, inputs, training=True):
        x = tf.cast(inputs, tf.float32)
        for dense, norm in self.hidden_layers:
            x = dense(x)
            if self.batchnorm:
                x = norm(x, training=training)
            x = tf.keras.activations.tanh(x)
        return self.output_layer(x)
