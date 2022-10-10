import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def read_data_2d():
    samples = pd.read_csv('2d_samples.csv', header=None).values.transpose()
    labels = pd.read_csv('2d_labels.csv', header=None).values[0]

    # print(samples.shape)
    # print(labels.shape)

    return samples, labels

def read_data_3d():
    samples = pd.read_csv('3d_samples.csv', header=None).values.transpose()
    labels = pd.read_csv('3d_labels.csv', header=None).values[0]

    return samples, labels

def draw_raw_data_2d():
    samples, labels = read_data_2d()
    plt.scatter(samples[0][0], samples[0][1], color='skyblue',\
        marker='o', label='L=1')
    plt.scatter(samples[4][0], samples[4][1], color='darkorange',\
        marker='+', label='L=0')

    for i in range(0, labels.size):
        if labels[i] == 1:
            plt.scatter(samples[i][0], samples[i][1], color='skyblue', marker='o')
        else:
            plt.scatter(samples[i][0], samples[i][1], color='darkorange', marker='+')

    plt.title("2D Data Distribution")
    
    plt.legend()
    plt.show()

def draw_raw_data_3d():
    samples, labels = read_data_3d()
    figure = plt.figure()
    axes = Axes3D(figure)
    
    axes.scatter(samples[4][0], samples[4][1], samples[4][2], color='lightcoral',\
        marker='o', label='L=1')
    axes.scatter(samples[2][0], samples[2][1], samples[2][2], color='turquoise',\
        marker='+', label='L=2')
    axes.scatter(samples[0][0], samples[0][1], samples[0][2], color='mediumpurple',\
        marker='2', label='L=3')

    for i in range(0, labels.size):
        if labels[i] == 1:
            axes.scatter(samples[i][0], samples[i][1], samples[i][2],\
                color='lightcoral',marker='o')
        elif labels[i] == 2:
            axes.scatter(samples[i][0], samples[i][1], samples[i][2],\
                color='turquoise',marker='+')
        else:
            axes.scatter(samples[i][0], samples[i][1], samples[i][2],\
                color='mediumpurple',marker='2')
    
    plt.legend()
    plt.show()