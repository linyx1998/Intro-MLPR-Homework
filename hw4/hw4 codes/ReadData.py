import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

def read_data(N):
    samples = pd.read_csv(str(N)+'_samples.csv', header=None).values.transpose()
    labels = pd.read_csv(str(N)+'_labels.csv', header=None).values[0]

    return samples, labels

def read_data_2(N):
    samples = pd.read_csv(str(N)+'_samples_2.csv', header=None).values.transpose()
    labels = pd.read_csv(str(N)+'_labels_2.csv', header=None).values[0]

    return samples, labels

def draw_raw_data(N):
    samples, labels = read_data(N)
    figure = plt.figure()
    axes = Axes3D(figure)
    
    markers = ('+','x','o','^')
    colors = ('darkorange','skyblue','lightcoral','mediumpurple')
    plt.title('Dataset with '+str(N)+' samples',loc='left')
    
    for idx,cl in enumerate(np.unique(labels)):
        axes.scatter(samples[labels==cl,0],
            samples[labels==cl,1],
            samples[labels==cl,2],
            alpha=0.6,
            c=colors[idx],
            marker = markers[idx],
            label="Label "+str(cl))

    plt.legend()
    plt.show()

def draw_raw_data_2(N):
    samples, labels = read_data_2(N)
    
    markers = ('+','x','o','^')
    colors = ('darkorange','skyblue','lightcoral','mediumpurple')
    
    for idx,cl in enumerate(np.unique(labels)):
        plt.scatter(samples[labels==cl,0],
            samples[labels==cl,1],
            alpha=0.8,
            c=colors[idx],
            marker = markers[idx],
            label="Label "+str(cl))

    plt.title('Dataset with '+str(N)+' samples')
    plt.legend()
    plt.show()