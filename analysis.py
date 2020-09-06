import numpy as np
from glob import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = list(glob('*.npy'))
source_d = 'known_mnist.npy'
target_d = [d for d in dataset if d != 'known_mnist.npy']


dataset = ['mnist', 'svhn', 'syn', 'usps', 'mnist_m']

source = np.load(source_d)
targets = [np.load(d) for d in target_d]

pca = PCA(n_components=1024)
pca.fit(source)
emb = pca.transform(source)

fig, axes = plt.subplots()
t = np.arange(0, 1024)
std = emb.std(axis=0)
axes.plot(t, (emb / std).std(axis=0), label='known_mnist')

for i, x in enumerate(targets):
    x = pca.transform(x)
    axes.plot(t, (x / std).std(axis=0), label=target_d[i])

axes.set_ylim(0, 4)
fig.legend()
fig.tight_layout()
plt.savefig('pca_variance.png')
