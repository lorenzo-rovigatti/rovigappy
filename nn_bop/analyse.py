#!/usr/bin/python3

import baggianalysis as ba
import numpy as np
import sys

import autoencoder as ae

COLORS = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "violet", "brown", "pink", "black", "grey"]

class MyParser(ba.BaseParser):
    def __init__(self):
        pass
    
    def parse(self, conf):
        syst = ba.System()
        
        with open(conf) as f:
            line = f.readline()
            spl = line.split()
            syst.time = int(spl[0])
            syst.box = [float(x) for x in spl[2:]]
            
            for line in f.readlines():
                pos = [float(x) for x in line.split()]
                particle = ba.Particle("0", pos, [0., 0., 0.])
                syst.add_particle(particle)
            
        return syst
            
            
def print_cogli1_conf(system, colors, filename):
    with open(filename, "w") as output:
        print(".Box:%lf,%lf,%lf" % tuple(system.box), file=output)
        for i, p in enumerate(system.particles()):
            print("%lf %lf %lf @ 0.5 C[%s]" % (p.position[0], p.position[1], p.position[2], colors[i]), file=output)
            
    
if len(sys.argv) < 2:
    print("Usage is %s input" % sys.argv[0], file=sys.stderr)
    exit(0)

input_filename = sys.argv[1]

parser = MyParser()
syst = parser.parse(input_filename)

# COMPUTE THE BOND ORDER PARAMETERS

nf = ba.SANNFinder(2.5, ba.SANNFinder.SYMMETRISE_BY_REMOVING)
nf.set_neighbours(syst.particles(), syst.box)

bop_obs = ba.BondOrderParameters({1, 2, 3, 4, 5, 6, 7, 8})
bops = np.array(bop_obs.compute(syst))

# ENCODE THE BOPS THROUGH UNSUPERVISED LEARNING

import tensorflow as tf

batch_size = 32
epochs = 10
learning_rate = 1e-2
momentum = 9e-1
code_dim = 2
original_dim = bops.shape[1]
hidden_dim = original_dim * 10
weight_lambda = 1e-5

training_dataset = tf.data.Dataset.from_tensor_slices(bops).batch(batch_size)

autoencoder = ae.Autoencoder(original_dim=original_dim, code_dim=code_dim, hidden_dim=hidden_dim, weight_lambda=weight_lambda)
opt = tf.optimizers.Adam(learning_rate=1e-3)

with open("loss_%s" % input_filename, "w") as loss_file:
    real_step = 0
    for epoch in range(epochs):
        for step, batch_features in enumerate(training_dataset):
            ae.train(ae.loss, autoencoder, opt, batch_features)
            loss_value = ae.loss(autoencoder, batch_features)
            real_step += 1
            print("%d %f" % (real_step, loss_value), file=loss_file)
        
encoded_bops = autoencoder.encoder(tf.constant(bops))

# FIND THE GAUSSIAN MIXTURE MODEL THAT BEST APPROXIMATES THE DATA THROUGH A BIC CRITERIUM

from sklearn import mixture
import itertools

lowest_bic = np.infty
bic = []
n_components_range = range(1, 10)
for n_components in n_components_range:
    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type="full")
    gmm.fit(encoded_bops)
    bic.append(gmm.bic(encoded_bops))
    # we choose the GMM with the lowest bic
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

np.savetxt("bic_%s" % input_filename, list(zip(n_components_range, bic)))

clf = best_gmm
cluster_probabilities = np.array(clf.predict_proba(encoded_bops))

def entropy(prob):
    return -np.sum(prob * np.log(prob))

def reduce_prob(prob_Y, to_reduce):
    # copy the array
    new_prob_Y = np.array(prob_Y)
    # sum the columns relative to the two clusters
    new_prob_Y[:,to_reduce[0]] += new_prob_Y[:,to_reduce[1]]
    # remove the column relative to the second element of the pair
    return np.delete(new_prob_Y, to_reduce[1], axis=1)


Ng = cluster_probabilities.shape[1]

print("The number of initial gaussians is %d" % Ng)

results = {}
results[Ng] = (np.copy(cluster_probabilities), entropy(cluster_probabilities))

# we reduce the number of gaussians by using the method of Baudry et al (2010)
while Ng > 1:
    clusters = list(range(Ng))

    min_entropy = np.infty
    # loop over all possible cluster pairs
    for to_reduce in itertools.combinations(clusters, 2):
        # sum the probabilities associated to the two clusters to be merged
        new_probabilities = reduce_prob(results[Ng][0], to_reduce)
        # calculate the new entropy
        new_entropy = entropy(new_probabilities)
        # keep track of the pair whose merging yields the smallest entropy
        if new_entropy < min_entropy:
            min_entropy = new_entropy
            min_probabilities = np.copy(new_probabilities)

    Ng -= 1
    results[Ng] = (min_probabilities, min_entropy)
    
with open("baudry_%s" % input_filename, "w") as baudry_file:
    for Ng in sorted(results.keys()):
        print(Ng, results[Ng][1], file=baudry_file)
        colors = list(map(lambda x: COLORS[np.argmax(x)], results[Ng][0]))
        print_cogli1_conf(syst, colors, "gauss_%d_%s" % (Ng, input_filename))
        