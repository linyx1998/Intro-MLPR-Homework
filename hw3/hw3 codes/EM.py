import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm

def read_data(name):
    samples = pd.read_csv(name+'_samples.csv', header=None).values.transpose()
    labels = pd.read_csv(name+'_labels.csv', header=None).values[0]

    # print(samples.shape)
    # print(labels.shape)

    return samples, labels

def EM(name):
    samples, labels = read_data(name)
    samples1 = []
    samples0 = []
    for i in range(len(labels)):
        if labels[i]==0:
            samples0.append(samples[i])
        else:
            samples1.append(samples[i])
    samples1 = np.array(samples1)
    samples0 = np.array(samples0)
    weight1, mean1, cov1 = parameter_estimate(samples1, 1, 1)
    weight0, mean0, cov0 = parameter_estimate(samples0, 2, 0)
    return mean0[0], mean0[1], mean1, cov0[0], cov0[1],\
        cov1, weight0[0], weight0[1]


def parameter_estimate(samples, components, label):
    clst = mixture.GaussianMixture(n_components=components)
    clst.fit(samples)
    x = np.linspace(-4., 12.)
    y = np.linspace(-5., 12.)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clst.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=100.0),
                    levels=np.logspace(0, 1, 10))
    # CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(samples[:, 0], samples[:, 1])

    plt.title('Data Distribution of Label '+str(label))
    plt.axis('tight')
    plt.show()
    return clst.weights_, clst.means_, clst.covariances_

def class_prior(name):
    samples, labels = read_data(name)
    amount = len(labels)
    cp1 = labels.tolist().count(1)/amount
    cp0 = labels.tolist().count(0)/amount
    print("Label=1:", cp1)
    print("Label=0:",cp0)
    return cp1, cp0

