import tensorflow as tf
import tensorflow.keras as keras
from tensorflow import function
from tensorflow import GradientTape 
from tensorflow import sqrt 
from tensorflow import abs 
from tensorflow import reduce_mean 
from tensorflow import ones_like 
from tensorflow import zeros_like 
from tensorflow import convert_to_tensor
from tensorflow import float32
from tensorflow import data as tfdata
from tensorflow import config as tfconfig
from tensorflow import nn

from joblib import dump, load
import pandas as pd
import numpy as np
from tqdm import tqdm, trange


def construct_network(model, n_layers, hidden_units, output_units, net_type='GRU'):
    if net_type=='GRU':
        for i in range(n_layers):
            model.add(keras.layers.GRU(units=hidden_units,
                                       return_sequences=True,
                                       name=f'GRU_{i + 1}'))
    else:
        for i in range(n_layers):
            model.add(keras.layers.LSTM(units=hidden_units,
                                        return_sequences=True,
                                        name=f'LSTM_{i + 1}'))

    model.add(keras.layers.Dense(units=output_units,
                                 activation='sigmoid',
                                 name='OUT'))
    return model


def unpack(model, training_config, weights):
    restored_model = keras.layers.deserialize(model)
    if training_config is not None:
        restored_model.compile(**keras.saving.saving_utils.compile_args_from_training_config(training_config))
    restored_model.set_weights(weights)
    return restored_model


def make_keras_picklable():
    def __reduce__(self):
        _metadata = keras.saving.saving_utils.model_metadata(self)
        training_config = _metadata.get("training_config", None)
        model = keras.layers.serialize(self)
        model_weights = self.get_weights()
        return (unpack, (model, training_config, model_weights))

    cls = keras.Model
    cls.__reduce__=__reduce__


class GanModel():
    def __init__(self, model_parameters):
        gpu_devices = tfconfig.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            try:
                tfconfig.experimental.set_memory_growth(gpu_devices[0], True)
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                pass

        self._model_parameters = model_parameters
        [self.batch_size,
         self.lr,
         self.beta_1,
         self.beta_2,
         self.noise_dim,
         self.data_dim,
         self.layers_dim] = model_parameters
        self.define_gan()

    def __call__(self, inputs, **kwargs):
        return self.model(inputs=inputs, **kwargs)

    def define_gan(self):
        raise NotImplementedError

    @property
    def trainable_variables(self, network):
        return network.trainable_variables

    @property
    def model_parameters(self):
        return self._model_parameters

    @property
    def model_name(self):
        return self.__class__.__name__

    def train(self, data, train_arguments):
        raise NotImplementedError

    def sample(self, n_samples):
        steps = n_samples // self.batch_size + 1
        data = []
        for _ in tqdm.trange(steps, desc='Synthetic data generation'):
            z = tf.random.uniform([self.batch_size, self.noise_dim])
            records = tf.make_ndarray(tf.make_tensor_proto(self.generator(z, training=False)))
            data.append(pd.DataFrame(records))
        return pd.concat(data)

    def save(self, path):
        make_keras_picklable()
        try:
            dump(self, path)
        except:
            raise Exception('Please provide a valid path to save the model.')

    @classmethod
    def load(cls, path):
        gpu_devices = tf.config.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            try:
                tfconfig.experimental.set_memory_growth(gpu_devices[0], True)
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                pass
        synth = load(path)
        return synth


class TimeGAN(GanModel):
    def __init__(self, model_parameters, hidden_dim, seq_len, n_seq, gamma):
        self.seq_len=seq_len
        self.n_seq=n_seq
        self.hidden_dim=hidden_dim
        self.gamma=gamma
        super().__init__(model_parameters)

    def define_gan(self):
        self.generator_aux=Generator(self.hidden_dim).build(input_shape=(self.seq_len, self.n_seq))
        self.supervisor=Supervisor(self.hidden_dim).build(input_shape=(self.hidden_dim, self.hidden_dim))
        self.discriminator=Discriminator(self.hidden_dim).build(input_shape=(self.hidden_dim, self.hidden_dim))
        self.recovery = Recovery(self.hidden_dim, self.n_seq).build(input_shape=(self.hidden_dim, self.hidden_dim))
        self.embedder = Embedder(self.hidden_dim).build(input_shape=(self.seq_len, self.n_seq))

        X = keras.Input(shape=[self.seq_len, self.n_seq], batch_size=self.batch_size, name='RealData')
        Z = keras.Input(shape=[self.seq_len, self.n_seq], batch_size=self.batch_size, name='RandomNoise')

        # Building the AutoEncoder
        H = self.embedder(X)
        X_tilde = self.recovery(H)

        self.autoencoder = keras.Model(inputs=X, outputs=X_tilde)

        # Adversarial Supervise Architecture
        E_Hat = self.generator_aux(Z)
        H_hat = self.supervisor(E_Hat)
        Y_fake = self.discriminator(H_hat)

        self.adversarial_supervised = keras.Model(inputs=Z,
                                       outputs=Y_fake,
                                       name='AdversarialSupervised')

        # Adversarial architecture in latent space
        Y_fake_e = self.discriminator(E_Hat)

        self.adversarial_embedded = keras.Model(inputs=Z,
                                    outputs=Y_fake_e,
                                    name='AdversarialEmbedded')
        # Synthetic data generation
        X_hat = self.recovery(H_hat)
        self.generator = keras.Model(inputs=Z,
                            outputs=X_hat,
                            name='FinalGenerator')

        # Final discriminator model
        Y_real = self.discriminator(H)
        self.discriminator_model = keras.Model(inputs=X,
                                         outputs=Y_real,
                                         name="RealDiscriminator")

        # Define the loss functions
        self._mse=keras.losses.MeanSquaredError()
        self._bce=keras.losses.BinaryCrossentropy()


    @function
    def train_autoencoder(self, x, opt):
        with GradientTape() as tape:
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x, x_tilde)
            e_loss_0 = 10 * sqrt(embedding_loss_t0)

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)

    @function
    def train_supervisor(self, x, opt):
        with GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            g_loss_s = self._mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

        var_list = self.supervisor.trainable_variables + self.generator.trainable_variables
        gradients = tape.gradient(g_loss_s, var_list)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
        opt.apply_gradients(apply_grads)
        return g_loss_s

    @function
    def train_embedder(self,x, opt):
        with GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self._mse(x, x_tilde)
            e_loss = 10 * sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return sqrt(embedding_loss_t0)

    def discriminator_loss(self, x, z):
        y_real = self.discriminator_model(x)
        discriminator_loss_real = self._bce(y_true=ones_like(y_real),
                                            y_pred=y_real)

        y_fake = self.adversarial_supervised(z)
        discriminator_loss_fake = self._bce(y_true=zeros_like(y_fake),
                                            y_pred=y_fake)

        y_fake_e = self.adversarial_embedded(z)
        discriminator_loss_fake_e = self._bce(y_true=zeros_like(y_fake_e),
                                              y_pred=y_fake_e)
        return (discriminator_loss_real +
                discriminator_loss_fake +
                self.gamma * discriminator_loss_fake_e)

    @staticmethod
    def calc_generator_moments_loss(y_true, y_pred):
        y_true_mean, y_true_var = nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = nn.moments(x=y_pred, axes=[0])
        g_loss_mean = reduce_mean(abs(y_true_mean - y_pred_mean))
        g_loss_var = reduce_mean(abs(sqrt(y_true_var + 1e-6) - sqrt(y_pred_var + 1e-6)))
        return g_loss_mean + g_loss_var

    @function
    def train_generator(self, x, z, opt):
        with GradientTape() as tape:
            y_fake = self.adversarial_supervised(z)
            generator_loss_unsupervised = self._bce(y_true=ones_like(y_fake),
                                                    y_pred=y_fake)

            y_fake_e = self.adversarial_embedded(z)
            generator_loss_unsupervised_e = self._bce(y_true=ones_like(y_fake_e),
                                                      y_pred=y_fake_e)
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self._mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

            x_hat = self.generator(z)
            generator_moment_loss = self.calc_generator_moments_loss(x, x_hat)

            generator_loss = (generator_loss_unsupervised +
                              generator_loss_unsupervised_e +
                              100 * sqrt(generator_loss_supervised) +
                              100 * generator_moment_loss)

        var_list = self.generator_aux.trainable_variables + self.supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss

    @function
    def train_discriminator(self, x, z, opt):
        with GradientTape() as tape:
            discriminator_loss = self.discriminator_loss(x, z)

        var_list = self.discriminator.trainable_variables
        gradients = tape.gradient(discriminator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return discriminator_loss

    def get_batch_data(self, data, n_windows):
        data = convert_to_tensor(data, dtype=float32)
        return iter(tfdata.Dataset.from_tensor_slices(data)
                                .shuffle(buffer_size=n_windows)
                                .batch(self.batch_size).repeat())

    def _generate_noise(self):
        while True:
            yield np.random.uniform(low=0, high=1, size=(self.seq_len, self.n_seq))

    def get_batch_noise(self):
        return iter(tfdata.Dataset.from_generator(self._generate_noise, output_types=float32)
                                .batch(self.batch_size)
                                .repeat())

    def train(self, data, train_steps):
        ## Embedding network training
        autoencoder_opt = keras.optimizers.Adam(learning_rate=self.lr)
        for _ in tqdm(range(train_steps), desc='Emddeding network training'):
            X_ = next(self.get_batch_data(data, n_windows=len(data)))
            step_e_loss_t0 = self.train_autoencoder(X_, autoencoder_opt)

        ## Supervised Network training
        supervisor_opt = keras.optimizers.Adam(learning_rate=self.lr)
        for _ in tqdm(range(train_steps), desc='Supervised network training'):
            X_ = next(self.get_batch_data(data, n_windows=len(data)))
            step_g_loss_s = self.train_supervisor(X_, supervisor_opt)

        ## Joint training
        generator_opt = keras.optimizers.Adam(learning_rate=self.lr)
        embedder_opt = keras.optimizers.Adam(learning_rate=self.lr)
        discriminator_opt = keras.optimizers.Adam(learning_rate=self.lr)

        step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
        for _ in tqdm(range(train_steps), desc='Joint networks training'):

            #Train the generator (k times as often as the discriminator)
            # Here k=2
            for _ in range(2):
                X_ = next(self.get_batch_data(data, n_windows=len(data)))
                Z_ = next(self.get_batch_noise())

                # Train the generator
                step_g_loss_u, step_g_loss_s, step_g_loss_v = self.train_generator(X_, Z_, generator_opt)

                # Train the embedder
                step_e_loss_t0 = self.train_embedder(X_, embedder_opt)

            X_ = next(self.get_batch_data(data, n_windows=len(data)))
            Z_ = next(self.get_batch_noise())
            step_d_loss = self.discriminator_loss(X_, Z_)
            if step_d_loss > 0.15:
                step_d_loss = self.train_discriminator(X_, Z_, discriminator_opt)

    def sample(self, n_samples):
        steps = n_samples // self.batch_size + 1
        data = []
        for _ in trange(steps, desc='Synthetic data generation'):
            Z_ = next(self.get_batch_noise())
            records = self.generator(Z_)
            data.append(records)
        return np.array(np.vstack(data))


class Generator(keras.Model):
    def __init__(self, hidden_dim, net_type='GRU'):
        self.hidden_dim = hidden_dim
        self.net_type = net_type

    def build(self, input_shape):
        model = keras.Sequential(name='Generator')
        model = construct_network(model,
                                  n_layers=3,
                                  hidden_units=self.hidden_dim,
                                  output_units=self.hidden_dim,
                                  net_type=self.net_type)
        return model


class Discriminator(keras.Model):
    def __init__(self, hidden_dim, net_type='GRU'):
        self.hidden_dim = hidden_dim
        self.net_type=net_type

    def build(self, input_shape):
        model = keras.Sequential(name='Discriminator')
        model = construct_network(model,
                                  n_layers=3,
                                  hidden_units=self.hidden_dim,
                                  output_units=1,
                                  net_type=self.net_type)
        return model


class Recovery(keras.Model):
    def __init__(self, hidden_dim, n_seq):
        self.hidden_dim=hidden_dim
        self.n_seq=n_seq
        return

    def build(self, input_shape):
        recovery = keras.Sequential(name='Recovery')
        recovery = construct_network(recovery,
                                     n_layers=3,
                                     hidden_units=self.hidden_dim,
                                     output_units=self.n_seq)
        return recovery


class Embedder(keras.Model):
    def __init__(self, hidden_dim):
        self.hidden_dim=hidden_dim
        return

    def build(self, input_shape):
        embedder = keras.Sequential(name='Embedder')
        embedder = construct_network(embedder,
                                     n_layers=3,
                                     hidden_units=self.hidden_dim,
                                     output_units=self.hidden_dim)
        return embedder


class Supervisor(keras.Model):
    def __init__(self, hidden_dim):
        self.hidden_dim=hidden_dim

    def build(self, input_shape):
        model = keras.Sequential(name='Supervisor')
        model = construct_network(model,
                                  n_layers=2,
                                  hidden_units=self.hidden_dim,
                                  output_units=self.hidden_dim)
        return model