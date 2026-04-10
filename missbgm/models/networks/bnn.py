"""Bayesian network modules retained for optional MissBGM generator support."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp


class BayesianVariationalNet(tf.keras.Model):
    """Bayesian generator with diagonal predictive covariance."""

    def __init__(self, input_dim, output_dim, model_name, nb_units=None):
        super(BayesianVariationalNet, self).__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.model_name = model_name
        self.nb_units = list(nb_units or [256, 256, 256])
        self.hidden_layers = []
        self.norm_layer = tf.keras.layers.BatchNormalization()

        kernel_prior_fn = (
            lambda dtype, shape, name, trainable, add_variable_fn: tfp.distributions.Independent(
                tfp.distributions.Normal(loc=tf.zeros(shape, dtype=dtype), scale=0.1),
                reinterpreted_batch_ndims=len(shape),
            )
        )

        for units in self.nb_units:
            self.hidden_layers.append(
                tfp.layers.DenseFlipout(
                    units=units,
                    activation=None,
                    kernel_prior_fn=kernel_prior_fn,
                    bias_prior_fn=kernel_prior_fn,
                )
            )
        self.mean_layer = tfp.layers.DenseFlipout(
            units=self.output_dim,
            activation=None,
            kernel_prior_fn=kernel_prior_fn,
            bias_prior_fn=kernel_prior_fn,
        )
        self.var_layer = tfp.layers.DenseFlipout(
            units=self.output_dim,
            activation=None,
            kernel_prior_fn=kernel_prior_fn,
            bias_prior_fn=kernel_prior_fn,
        )

    def call(self, inputs, eps=1e-6, training=True):
        x = self.norm_layer(tf.cast(inputs, tf.float32), training=training)
        for layer in self.hidden_layers:
            x = layer(x, training=training)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        mean = self.mean_layer(x, training=training)
        var = tf.nn.softplus(self.var_layer(x, training=training)) + eps
        return mean, var

    def reparameterize(self, mean, var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.sqrt(var) + mean
