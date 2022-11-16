import numpy as np
import random
import csv

def generate_data_2d(N, name):
    rand_label = []
    for i in range(N):
        rd = random.uniform(0, 1)
        if rd <= 0.6:
            rand_label.append(0)
        else:
            rand_label.append(1)
    labels = np.array(rand_label)

    m01 = (5, 0)
    m02 = (0, 4)
    m1 = (3, 2)
    C01 = [[4,0], [0,2]]
    C02 = [[1,0], [0,3]]
    C1 = [[2,0], [0,2]]

    rand_sample = []
    for i in range(N):
        if rand_label[i] == 1:
            sample = np.random.multivariate_normal(m1, C1)
            rand_sample.append(sample.tolist())
        else:
            if random.uniform(0, 1)>=0.5:
                sample = np.random.multivariate_normal(m01, C01)
                rand_sample.append(sample.tolist())
            else:
                sample = np.random.multivariate_normal(m02, C02)
                rand_sample.append(sample.tolist())
    samples = np.array(rand_sample)
    # print(rand_sample)

    with open(name+"_labels.csv", 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerow(labels)
    with open(name+"_samples.csv", 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerows(samples.transpose())

    return samples, labels
