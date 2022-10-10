import numpy as np
from ReadData import *

def lda_classify(samples, labels, tau):
    w = gen_eigendecomposition(samples, labels)

    pro_samples = []
    for i in range(samples.shape[0]):
        pro_samples.append(np.dot(w.transpose(), samples[i]))
    pro_samples = np.array(pro_samples).ravel()
    # print(pro_samples)

    predict = []
    for i in range(pro_samples.size):
        if pro_samples[i] <= tau:
            predict.append(1)
        else:
            predict.append(0)
    # print(predict)

    return np.array(predict)

def gen_eigendecomposition(samples, labels):
    sample0 = []
    sample1 = []
    for i in range(samples.shape[0]):
        if labels[i] == 1:
            sample1.append(samples[i])
        else:
            sample0.append(samples[i])

    sample0 = np.matrix(sample0)
    sample1 = np.matrix(sample1)

    mu0 = np.mean(sample0, axis=0).transpose()
    mu1 = np.mean(sample1, axis=0).transpose()
    cov0 = np.cov(sample0.transpose())
    cov1 = np.cov(sample1.transpose())
    # print(mu0, mu1)
    # print(cov0, cov1)

    Sw = np.matrix(cov0 + cov1)
    Sb = np.dot((mu0 - mu1), (mu0-mu1).transpose())
    SwISb = np.dot(Sw.I, Sb)
    eig = np.linalg.eig(SwISb)

    for i in range(eig[0].size):
        if eig[0][i] == max(eig[0]):
            w = (eig[1].transpose()[i]).transpose()
            # print(w)
            return w
    return -1

# lda_classify(read_data_2d()[0], read_data_2d()[1], 1)