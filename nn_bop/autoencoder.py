'''
Created on 8 ago 2019

@author: lorenzo
'''

# adapted from https://medium.com/@abien.agarap/implementing-an-autoencoder-in-tensorflow-2-0-5e86126e9f7 (code here: https://gist.github.com/AFAgarap/326af55e36be0529c507f1599f88c06e)

import numpy as np
import tensorflow as tf

class Autoencoder(tf.keras.Model):

    def __init__(self, original_dim, code_dim, hidden_dim, weight_lambda):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
                                            tf.keras.layers.Dense(units=hidden_dim, activation=tf.nn.tanh),
                                            tf.keras.layers.Dense(units=code_dim, activation=None)
                                        ])
        self.decoder = tf.keras.Sequential([
                                            tf.keras.layers.Dense(units=hidden_dim, activation=tf.nn.tanh),
                                            tf.keras.layers.Dense(units=original_dim, activation=None)
                                        ])
        self.weight_lambda = weight_lambda
    
    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed
    
    def kernel(self):
        kernel = filter(lambda x: "kernel" in x.name, self.trainable_weights)
        kernel = tf.concat([tf.reshape(x, [-1]) for x in kernel], 0)
        return kernel 


def loss(model, original):
    reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original))) + model.weight_lambda * tf.reduce_sum(tf.square(model.kernel()))
    return reconstruction_error

  
def train(loss, model, opt, original):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, original), model.trainable_variables)
    gradient_variables = zip(gradients, model.trainable_variables)
    opt.apply_gradients(gradient_variables)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage is %s input" % sys.argv[0], file=sys.stderr)
        exit(1)

    np.random.seed(1)
    tf.random.set_seed(1)
    batch_size = 32
    epochs = 10
    learning_rate = 1e-2
    momentum = 9e-1
    original_dim = 8
    code_dim = 2
    hidden_dim = original_dim * 10
    weight_lambda = 1e-5
    
    (training_features, _), _ = tf.keras.datasets.mnist.load_data()
    training_features = training_features / np.max(training_features)
    training_features = training_features.reshape(training_features.shape[0],
                                                  training_features.shape[1] * training_features.shape[2]).astype(np.float32)
    features = np.loadtxt(sys.argv[1], usecols=(1, 2, 3, 4, 5, 6, 7, 8))
    training_dataset = tf.data.Dataset.from_tensor_slices(features).batch(batch_size)
    
    autoencoder = Autoencoder(original_dim=original_dim, code_dim=code_dim, hidden_dim=hidden_dim, weight_lambda=weight_lambda)
    # opt = tf.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    opt = tf.optimizers.Adam(learning_rate=1e-3)
    
    writer = tf.summary.create_file_writer('tmp')
    
    with writer.as_default():
        with tf.summary.record_if(True):
            real_step = 0
            for epoch in range(epochs):
                for step, batch_features in enumerate(training_dataset):
                    train(loss, autoencoder, opt, batch_features)
                    loss_value = loss(autoencoder, batch_features)
                    tf.summary.scalar('loss', loss_value, step=real_step)
                    real_step += 1
    
    np.savetxt("output.dat", autoencoder.encoder(tf.constant(features)))
