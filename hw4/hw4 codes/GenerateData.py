import numpy as np
import random
import csv

def generate_data(N):
    rand_label = []
    # here we have C=4
    for i in range(N):
        rd = random.uniform(0, 1)
        if rd <= 0.25:
            rand_label.append(1)
        elif rd <= 0.5:
            rand_label.append(2)
        elif rd <= 0.75:
            rand_label.append(3)
        else:
            rand_label.append(4)
    labels = np.array(rand_label)
    
    mean1 = (-5,3,-2)
    mean2 = (2,5,3)
    mean3 = (5,-2,-2)
    mean4 = (0,-5,-3)

    cov1 = [[10,3,-2], [3,5,0], [-2,0,10]]
    cov2 = [[5,1,0], [1,10,0], [0,0,5]]
    cov3 = [[10,-2,0], [-2,5,0], [0,0,10]]
    cov4 = [[10,0,0], [0,10,5], [0,5,10]]

    rand_sample = []
    for i in range(N):
        if rand_label[i] == 1:
            sample = np.random.multivariate_normal(mean1, cov1)
            rand_sample.append(sample.tolist())
        elif rand_label[i] == 2:
            sample = np.random.multivariate_normal(mean2, cov2)
            rand_sample.append(sample.tolist())
        elif rand_label[i] == 3:
            sample = np.random.multivariate_normal(mean3, cov3)
            rand_sample.append(sample.tolist())
        else:
            sample = np.random.multivariate_normal(mean4, cov4)
            rand_sample.append(sample.tolist())
    samples = np.array(rand_sample)

    with open(str(N)+"_labels.csv", 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerow(labels)
    with open(str(N)+"_samples.csv", 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerows(samples.transpose())
    
    return samples, labels

def generate_data_2(N):
    rand_label = []
    # here we have C=4
    for i in range(N):
        rd = random.uniform(0, 1)
        if rd <= 0.22:
            rand_label.append(1) # 0.22
        elif rd <= 0.5:
            rand_label.append(2) # 0.28
        elif rd <= 0.74:
            rand_label.append(3) # 0.24
        else:
            rand_label.append(4) # 0.26
    labels = np.array(rand_label)

    mean1 = (-8,8)
    mean2 = (8,-8)
    mean3 = (-8,-8)
    mean4 = (8,8)

    cov1 = [[7,-1],[-1,7]]
    cov2 = [[8,0],[0,8]]
    cov3 = [[6,0],[0,6]]
    cov4 = [[9,2],[2,9]]

    rand_sample = []
    for i in range(N):
        if rand_label[i] == 1:
            sample = np.random.multivariate_normal(mean1, cov1)
            rand_sample.append(sample.tolist())
        elif rand_label[i] == 2:
            sample = np.random.multivariate_normal(mean2, cov2)
            rand_sample.append(sample.tolist())
        elif rand_label[i] == 3:
            sample = np.random.multivariate_normal(mean3, cov3)
            rand_sample.append(sample.tolist())
        else:
            sample = np.random.multivariate_normal(mean4, cov4)
            rand_sample.append(sample.tolist())
    samples = np.array(rand_sample)

    with open(str(N)+"_labels_2.csv", 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerow(labels)
    with open(str(N)+"_samples_2.csv", 'w', newline='') as file: 
        writer = csv.writer(file)
        writer.writerows(samples.transpose())
    
    return samples, labels



