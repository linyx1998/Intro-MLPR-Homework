import numpy as np
import random
import csv

def generate_data_2d(N):
    rand_label = []
    for i in range(N):
        rd = random.uniform(0, 1)
        if rd <= 0.65:
            rand_label.append(0)
        else:
            rand_label.append(1)
    labels = np.array(rand_label)

    m01 = (3, 0)
    m02 = (0, 3)
    m1 = (2, 2)
    C01 = [[2,0], [0,1]]
    C02 = [[1,0], [0,2]]
    C1 = [[1,0], [0,1]]

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

    with open("2d_labels.csv", 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerow(labels)
    with open("2d_samples.csv", 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerows(samples.transpose())

    return samples, labels

def generate_data_3d(N):
    rand_label = []
    for i in range(N):
        rd = random.uniform(0, 1)
        if rd <= 0.3:
            rand_label.append(1)
        elif rd >= 0.6:
            rand_label.append(3)
        else:
            rand_label.append(2)
    labels = np.array(rand_label)
    
    mean1 = (1,1,1)
    mean2 = (0,0,0)
    mean3 = (1,0,1)
    mean4 = (0,1,0)
    cov = [[2,0,0], [0,2,0], [0,0,2]]

    rand_sample = []
    for i in range(N):
        if rand_label[i] == 1:
            sample = np.random.multivariate_normal(mean1, cov)
            rand_sample.append(sample.tolist())
        elif rand_label[i] == 2:
            sample = np.random.multivariate_normal(mean2, cov)
            rand_sample.append(sample.tolist())
        else:
            if random.uniform(0, 1)>=0.5:
                sample = np.random.multivariate_normal(mean3, cov)
                rand_sample.append(sample.tolist())
            else:
                sample = np.random.multivariate_normal(mean4, cov)
                rand_sample.append(sample.tolist())
    samples = np.array(rand_sample)

    with open("3d_labels.csv", 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerow(labels)
    with open("3d_samples.csv", 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerows(samples.transpose())
    
    return samples, labels
