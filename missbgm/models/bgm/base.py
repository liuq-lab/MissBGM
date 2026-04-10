"""Core latent-variable model used by MissBGM."""

from __future__ import annotations

import datetime
import os

import dateutil.tz
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ...datasets import Base_sampler, Gaussian_sampler
from ..networks import (
    BaseFullyConnectedNet,
    BaseVariationalNet,
    BayesianVariationalNet,
    Discriminator,
)


class BGM(object):
    """Base Bayesian generative model used by MissBGM."""

    def __init__(self, params, timestamp=None, random_seed=None):
        super(BGM, self).__init__()
        self.params = params
        self.timestamp = timestamp

        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            tf.config.experimental.enable_op_determinism()

        generator_cls = BayesianVariationalNet if self.params["use_bnn"] else BaseVariationalNet
        self.g_net = generator_cls(
            input_dim=params["z_dim"],
            output_dim=params["x_dim"],
            model_name="g_net",
            nb_units=params["g_units"],
        )
        self.e_net = BaseFullyConnectedNet(
            input_dim=params["x_dim"],
            output_dim=params["z_dim"],
            model_name="e_net",
            nb_units=params["e_units"],
        )
        self.dz_net = Discriminator(
            input_dim=params["z_dim"],
            model_name="dz_net",
            nb_units=params["dz_units"],
        )
        self.dx_net = Discriminator(
            input_dim=params["x_dim"],
            model_name="dx_net",
            nb_units=params["dx_units"],
        )

        self.g_pre_optimizer = tf.keras.optimizers.Adam(params["lr"], beta_1=0.5, beta_2=0.9)
        self.d_pre_optimizer = tf.keras.optimizers.Adam(params["lr"], beta_1=0.5, beta_2=0.9)
        self.g_optimizer = tf.keras.optimizers.Adam(params["lr_theta"], beta_1=0.9, beta_2=0.99)
        self.posterior_optimizer = tf.keras.optimizers.Adam(params["lr_z"], beta_1=0.9, beta_2=0.99)
        self.z_sampler = Gaussian_sampler(mean=np.zeros(params["z_dim"]), sd=1.0)

        self.initialize_nets()
        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime("%Y%m%d_%H%M%S")

        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            params["output_dir"],
            params["dataset"],
            self.timestamp,
        )
        self.save_dir = "{}/results/{}/{}".format(
            params["output_dir"],
            params["dataset"],
            self.timestamp,
        )

        if self.params["save_model"]:
            os.makedirs(self.checkpoint_path, exist_ok=True)
        if self.params["save_res"]:
            os.makedirs(self.save_dir, exist_ok=True)

        self.ckpt = tf.train.Checkpoint(
            g_net=self.g_net,
            e_net=self.e_net,
            dz_net=self.dz_net,
            dx_net=self.dx_net,
            g_pre_optimizer=self.g_pre_optimizer,
            d_pre_optimizer=self.d_pre_optimizer,
            g_optimizer=self.g_optimizer,
            posterior_optimizer=self.posterior_optimizer,
        )
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=100)
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored.")

    def get_config(self):
        return {"params": self.params}

    def initialize_nets(self, print_summary=False):
        self.g_net(np.zeros((1, self.params["z_dim"]), dtype=np.float32))
        if print_summary:
            print(self.g_net.summary())

    @tf.function
    def train_disc_step(self, data_z, data_x):
        epsilon_z = tf.random.uniform([], minval=0.0, maxval=1.0)
        epsilon_x = tf.random.uniform([], minval=0.0, maxval=1.0)
        with tf.GradientTape(persistent=True) as disc_tape:
            with tf.GradientTape() as gpz_tape:
                data_z_ = self.e_net(data_x)
                data_z_hat = data_z * epsilon_z + data_z_ * (1.0 - epsilon_z)
                data_dz_hat = self.dz_net(data_z_hat)
            with tf.GradientTape() as gpx_tape:
                mu_x_, sigma_square_x_ = self.g_net(data_z)
                data_x_ = self.g_net.reparameterize(mu_x_, sigma_square_x_)
                data_x_hat = data_x * epsilon_x + data_x_ * (1.0 - epsilon_x)
                data_dx_hat = self.dx_net(data_x_hat)

            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            data_dx = self.dx_net(data_x)
            data_dz = self.dz_net(data_z)

            dz_loss = (
                tf.reduce_mean((0.9 * tf.ones_like(data_dz) - data_dz) ** 2)
                + tf.reduce_mean((0.1 * tf.ones_like(data_dz_) - data_dz_) ** 2)
            ) / 2.0
            dx_loss = (
                tf.reduce_mean((0.9 * tf.ones_like(data_dx) - data_dx) ** 2)
                + tf.reduce_mean((0.1 * tf.ones_like(data_dx_) - data_dx_) ** 2)
            ) / 2.0

            grad_z = gpz_tape.gradient(data_dz_hat, data_z_hat)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))

            grad_x = gpx_tape.gradient(data_dx_hat, data_x_hat)
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=1))
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))

            d_loss = dx_loss + dz_loss + self.params["gamma"] * (gpz_loss + gpx_loss)

        d_gradients = disc_tape.gradient(
            d_loss,
            self.dz_net.trainable_variables + self.dx_net.trainable_variables,
        )
        self.d_pre_optimizer.apply_gradients(
            zip(d_gradients, self.dz_net.trainable_variables + self.dx_net.trainable_variables)
        )
        return dz_loss, dx_loss, d_loss

    @tf.function
    def train_gen_step(self, data_z, data_x):
        with tf.GradientTape(persistent=True) as gen_tape:
            mu_x_, sigma_square_x_ = self.g_net(data_z)
            data_x_ = self.g_net.reparameterize(mu_x_, sigma_square_x_)
            reg_loss = tf.reduce_mean(tf.square(sigma_square_x_))
            data_z_ = self.e_net(data_x)

            data_z__ = self.e_net(data_x_)
            mu_x__, sigma_square_x__ = self.g_net(data_z_)
            data_x__ = self.g_net.reparameterize(mu_x__, sigma_square_x__)

            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)

            l2_loss_x = tf.reduce_mean((data_x - data_x__) ** 2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__) ** 2)
            g_loss_adv = tf.reduce_mean((0.9 * tf.ones_like(data_dx_) - data_dx_) ** 2)
            e_loss_adv = tf.reduce_mean((0.9 * tf.ones_like(data_dz_) - data_dz_) ** 2)
            g_e_loss = g_loss_adv + e_loss_adv + 10.0 * (l2_loss_x + l2_loss_z) + self.params["alpha"] * reg_loss

        g_e_gradients = gen_tape.gradient(
            g_e_loss,
            self.g_net.trainable_variables + self.e_net.trainable_variables,
        )
        self.g_pre_optimizer.apply_gradients(
            zip(g_e_gradients, self.g_net.trainable_variables + self.e_net.trainable_variables)
        )
        return g_loss_adv, e_loss_adv, l2_loss_z, l2_loss_x, reg_loss, g_e_loss

    def egm_init(self, data, egm_n_iter=10000, batch_size=32, egm_batches_per_eval=500, verbose=1):
        self.data_sampler = Base_sampler(x=data, y=data, v=data, batch_size=batch_size, normalize=False)
        print("EGM initialization starts...")
        for batch_iter in range(egm_n_iter + 1):
            for _ in range(self.params["g_d_freq"]):
                batch_x, _, _ = self.data_sampler.next_batch()
                batch_z = self.z_sampler.get_batch(batch_size)
                dz_loss, dx_loss, d_loss = self.train_disc_step(batch_z, batch_x)

            batch_x, _, _ = self.data_sampler.next_batch()
            batch_z = self.z_sampler.get_batch(batch_size)
            g_loss_adv, e_loss_adv, l2_loss_z, l2_loss_x, sigma_square_loss, g_e_loss = self.train_gen_step(
                batch_z,
                batch_x,
            )

            if batch_iter % egm_batches_per_eval == 0:
                if verbose:
                    print(
                        (
                            "EGM Iter [{}] g_loss_adv[{:.4f}] e_loss_adv[{:.4f}] "
                            "l2_z[{:.4f}] l2_x[{:.4f}] sd2[{:.4f}] g_e[{:.4f}] "
                            "dz[{:.4f}] dx[{:.4f}] d[{:.4f}]"
                        ).format(
                            batch_iter,
                            float(g_loss_adv),
                            float(e_loss_adv),
                            float(l2_loss_z),
                            float(l2_loss_x),
                            float(sigma_square_loss),
                            float(g_e_loss),
                            float(dz_loss),
                            float(dx_loss),
                            float(d_loss),
                        )
                    )
                    mse_x = self.evaluate(data=data, use_x_sd=False)
                    print("EGM reconstruction MSE:", float(mse_x))

                if self.params["save_model"]:
                    base_path = self.checkpoint_path + f"/weights_at_egm_init_{batch_iter}"
                    self.e_net.save_weights(f"{base_path}_encoder.weights.h5")
                    self.g_net.save_weights(f"{base_path}_generator.weights.h5")
        print("EGM initialization ends.")

    @tf.function
    def evaluate(self, data, data_z=None, use_x_sd=True):
        if data_z is None:
            data_z = self.e_net(data, training=False)

        mu_x, sigma_square_x = self.g_net(data_z, training=False)
        data_x_pred = self.g_net.reparameterize(mu_x, sigma_square_x) if use_x_sd else mu_x
        return tf.reduce_mean((data - data_x_pred) ** 2)

    @tf.function
    def generate(self, nb_samples=1000, use_x_sd=True):
        data_z = tf.random.normal(shape=(nb_samples, self.params["z_dim"]), mean=0.0, stddev=1.0)
        mu_x, sigma_square_x = self.g_net(data_z, training=False)
        data_x_gen = self.g_net.reparameterize(mu_x, sigma_square_x) if use_x_sd else mu_x
        return data_x_gen, sigma_square_x

    @tf.function
    def get_log_posterior(self, data_z, data_x, ind_x1=None, obs_mask=None):
        mu_x, sigma_square_x = self.g_net(data_z, training=False)

        if ind_x1 is None:
            loss_px_z = tf.reduce_sum(
                ((data_x - mu_x) ** 2) / (2.0 * sigma_square_x) + 0.5 * tf.math.log(sigma_square_x),
                axis=1,
            )
        else:
            data_x_cond = tf.gather(data_x, ind_x1, batch_dims=1)
            mu_x_cond = tf.gather(mu_x, ind_x1, batch_dims=1)
            sigma_square_x_cond = tf.gather(sigma_square_x, ind_x1, batch_dims=1)
            ll_term = ((data_x_cond - mu_x_cond) ** 2) / (2.0 * sigma_square_x_cond) + 0.5 * tf.math.log(
                sigma_square_x_cond
            )
            if obs_mask is not None:
                ll_term = ll_term * obs_mask
            loss_px_z = tf.reduce_sum(ll_term, axis=1)

        loss_prior_z = tf.reduce_sum(data_z ** 2, axis=1) / 2.0
        return -(loss_prior_z + loss_px_z)
