'''
Created on 8 ago 2019

@author: lorenzo
'''

# taken from https://medium.com/@abien.agarap/implementing-an-autoencoder-in-tensorflow-2-0-5e86126e9f7 (code here: https://gist.github.com/AFAgarap/326af55e36be0529c507f1599f88c06e)

import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.random.set_seed(1)
batch_size = 128
epochs = 100
learning_rate = 1e-2
momentum = 9e-1
intermediate_dim = 64
original_dim = 784

(training_features, _), _ = tf.keras.datasets.mnist.load_data()
training_features = training_features / np.max(training_features)
training_features = training_features.reshape(training_features.shape[0],
                                              training_features.shape[1] * training_features.shape[2]).astype(np.float32)
training_dataset = tf.data.Dataset.from_tensor_slices(training_features).batch(batch_size)


class Encoder(tf.keras.layers.Layer):

    def __init__(self, intermediate_dim):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
      
    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)


class Decoder(tf.keras.layers.Layer):

    def __init__(self, intermediate_dim, original_dim):
        super(Decoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=original_dim, activation=tf.nn.relu)
  
    def call(self, code):
        activation = self.hidden_layer(code)
        return self.output_layer(activation)

  
class Autoencoder(tf.keras.Model):

    def __init__(self, intermediate_dim, original_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(intermediate_dim=intermediate_dim)
        self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)
    
    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed


autoencoder = Autoencoder(intermediate_dim=intermediate_dim, original_dim=original_dim)
#opt = tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
opt = tf.optimizers.Adam(learning_rate=1e-3)


def loss(model, original):
    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
    return reconstruction_error

  
def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original), model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)


writer = tf.summary.create_file_writer('tmp')

with writer.as_default():
    with tf.summary.record_if(True):
        real_step = 0
        for epoch in range(epochs):
            for step, batch_features in enumerate(training_dataset):
                train(loss, autoencoder, opt, batch_features)
                loss_value = loss(autoencoder, batch_features)
                original = tf.reshape(batch_features, (batch_features.shape[0], 28, 28, 1))
                reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)), (batch_features.shape[0], 28, 28, 1))
                tf.summary.scalar('loss', loss_value, step=real_step)
                tf.summary.image('original', original, max_outputs=10, step=real_step)
                tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=real_step)
                real_step += 1
