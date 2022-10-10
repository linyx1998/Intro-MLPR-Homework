import numpy as np

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
    m01 = np.matrix([3, 0]).transpose()
    m02 = np.matrix([0, 3]).transpose()
    m1 = np.matrix([2, 2]).transpose()

    C01 = np.matrix([[2,0], [0,1]])
    C02 = np.matrix([[1,0], [0,2]])
    C1 = np.matrix([[1,0], [0,1]])

    ld_ratio = gaussian_pdf(sample, m1, C1)/\
    (0.5*(gaussian_pdf(sample, m01, C01) + gaussian_pdf(sample, m02, C02)))

    # print(ld_ratio)
    return ld_ratio

def gaussian_pdf(sample, mean, cov):
    sample = np.matrix(sample).transpose()

    g = 1/(pow(2*np.pi, sample.size/2)*pow(np.linalg.det(cov), 0.5))
    g = g*np.exp(-0.5*np.dot(np.dot((sample-mean).transpose(), cov.I),(sample-mean)))

    return np.float64(g)
