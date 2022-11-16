import numpy as np
from EM import *

def erm_classify(samples, gamma):
    predict = []
    for i in range(samples.shape[0]):
        sample = samples[i]
        if cal_likelihood_ratio(sample) >= gamma:
            predict.append(1)
        else:
            predict.append(0)

    # print(predict)
    return np.array(predict)

def cal_likelihood_ratio(sample):
    m01 = np.matrix([5, 0]).transpose()
    m02 = np.matrix([0, 4]).transpose()
    m1 = np.matrix([3, 2]).transpose()

    C01 = np.matrix([[4,0], [0,2]])
    C02 = np.matrix([[1,0], [0,3]])
    C1 = np.matrix([[2,0], [0,2]])

    ld_ratio = gaussian_pdf(sample, m1, C1)/\
    (0.5*(gaussian_pdf(sample, m01, C01) + gaussian_pdf(sample, m02, C02)))

    # print(ld_ratio)
    return ld_ratio

def gaussian_pdf(sample, mean, cov):
    sample = np.matrix(sample).transpose()

    g = 1/(pow(2*np.pi, sample.size/2)*pow(np.linalg.det(cov), 0.5))
    g = g*np.exp(-0.5*np.dot(np.dot((sample-mean).transpose(), cov.I),(sample-mean)))

    return np.float64(g)

def erm_classify_auto(samples, gamma, data_name):
    predict = []
    m01, m02, m1, C01, C02, C1, w1, w2 = EM(data_name)
    # print("m01:", m01)
    # print("m02:", m02)
    # print("m1:", m1)
    # print("C01:", C01)
    # print("C02:", C02)
    # print("C1:", C1)
    # print("w1:", w1)
    # print("w2:", w2)
    for i in range(samples.shape[0]):
        sample = samples[i]
        if cal_likelihood_ratio_auto(sample, \
            m01, m02, m1, C01, C02, C1, w1, w2) >= gamma:
            predict.append(1)
        else:
            predict.append(0)

    # print(predict)
    return np.array(predict)

def cal_likelihood_ratio_auto(sample, m01, m02, m1, C01, C02, C1, w1, w2):
    m01 = np.matrix(m01).transpose()
    m02 = np.matrix(m02).transpose()
    m1 = np.matrix(m1).transpose()

    C01 = np.matrix(C01)
    C02 = np.matrix(C02)
    C1 = np.matrix(C1)

    ld_ratio = gaussian_pdf(sample, m1, C1)/\
    (w1*gaussian_pdf(sample, m01, C01) + w2*gaussian_pdf(sample, m02, C02))

    # print(ld_ratio)
    return ld_ratio

