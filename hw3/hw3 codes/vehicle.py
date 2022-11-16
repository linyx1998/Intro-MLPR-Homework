import numpy as np
import math
import matplotlib.pyplot as plt

def exec_estimate():
    sample = generate_sample()
    for K in range(1,5):
        landmarks = generate_landmark(K) 
        ranges = generate_range(landmarks, sample)
        plot_estimate(sample, landmarks, ranges)


def plot_estimate(sample, landmarks, ranges):
    xx1,xx2 = np.meshgrid(
        np.linspace(-2, 2, int(4*100)).reshape(-1, 1),
        np.linspace(-2, 2, int(4*100)).reshape(-1, 1),
        )
    sample_set = np.c_[xx1.ravel(), xx2.ravel()]
    Z = map_estimate(sample_set, landmarks, ranges)
    estimate_index = np.argmin(Z)
    estimate_position = sample_set[estimate_index]

    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,25,alpha=0.45,cmap='GnBu_r')
    plt.colorbar()
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    plt.scatter(x=landmarks[:,0],
        y = landmarks[:,1],
        # alpha=0.6,
        c="royalblue",
        marker = 'o',
        label="landmarks")
    plt.scatter(x=sample[0],
        y = sample[1],
        s=80,
        c='darkorange',
        marker = '+',
        label="vehicle true position")
    plt.scatter(x=estimate_position[0],
        y = estimate_position[1],
        s=80,
        c='purple',
        marker = '*',
        label="vehicle estimated position")

    print(np.linalg.norm(estimate_position-sample))

    plt.legend(loc='upper left')
    plt.title("MAP Objective Function Contours (K="+str(landmarks.shape[0])+")")
    plt.show()


def generate_landmark(K):
    landmarks = []
    for i in range(K):
        theta = i*(2*np.pi)/K
        r = 1
        landmarks.append([r*math.cos(theta),\
            r * math.sin(theta)])
    return np.array(landmarks)

def generate_sample():
    theta = np.random.uniform(0, 2*np.pi)
    r = np.random.uniform(0, 1)
    return np.array([r*math.cos(theta),\
            r * math.sin(theta)])

def generate_range(landmarks, sample):
    ranges = []
    # print(landmarks.shape[0])
    for i in range(landmarks.shape[0]):
        di = np.linalg.norm(landmarks[i]-sample)
        ni = np.random.normal(0, 0.3)
        ri = di+ni
        while ri<=0:
            ni = np.random.normal(0, 0.3)
            ri = di+ni
        ranges.append(ri)
    return np.array(ranges)

def map_estimate(samples, landmarks, ranges):
    estimation = []
    for i in range(samples.shape[0]):
        es = np.power(samples[i][0],2)/np.power(0.25,2) + \
            np.power(samples[i][1],2)/np.power(0.25,2)
        for k in range(ranges.shape[0]):
            di = np.linalg.norm(landmarks[k]-samples[i])
            es += np.power((di-ranges[k]),2)/(2*np.power(0.3,2))
        estimation.append(es)
    return np.array(estimation)
