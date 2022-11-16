import matplotlib.pyplot as plt
import pandas as pd
from ERMClassifier import *
from matplotlib.colors import ListedColormap
import numpy as np

def read_data_2d(name):
    samples = pd.read_csv(name+'_samples.csv', header=None).values.transpose()
    labels = pd.read_csv(name+'_labels.csv', header=None).values[0]

    # print(samples.shape)
    # print(labels.shape)

    return samples, labels

def draw_raw_data_2d(name):
    samples, labels = read_data_2d(name)
    plt.scatter(samples[0][0], samples[0][1], color='skyblue',\
        marker='o', label='L=1')
    plt.scatter(samples[4][0], samples[4][1], color='darkorange',\
        marker='+', label='L=0')

    for i in range(0, labels.size):
        if labels[i] == 1:
            plt.scatter(samples[i][0], samples[i][1], color='skyblue', marker='o')
        else:
            plt.scatter(samples[i][0], samples[i][1], color='darkorange', marker='+')

    plt.title(name+" Data Distribution")
    
    plt.legend()
    plt.show()

def draw_data_boundary(name):
    samples, labels = read_data_2d(name)

    markers = ('+','x')
    colors = ('darkorange','skyblue')
    colors2 = ('orangered', 'lime')
    cmap = ListedColormap(colors[:len(np.unique(labels))])

    x1_min,x1_max = samples[:,0].min()-1,samples[:,0].max()+1
    x2_min,x2_max = samples[:,1].min()-1,samples[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,0.05),
                         np.arange(x2_min,x2_max,0.05))  
    Z = erm_classify(np.array([xx1.ravel(),xx2.ravel()]).T, 1.5)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    predict_label = erm_classify(samples, 1.5)
    decision = []
    for i in range(labels.size):
        decision.append(labels[i]==predict_label[i])
    decision = np.array(decision)

    # plot class samples
    # for idx,cl in enumerate(np.unique(labels)):
    #     plt.scatter(x=samples[labels==cl,0],
    #         y = samples[labels==cl,1],
    #         alpha=0.6,
    #         c=colors[idx],
    #         marker = markers[idx],
    #         label="L="+str(cl))
    for idx,cl in enumerate(np.unique(decision)):
        plt.scatter(x=samples[decision==cl,0],
            y = samples[decision==cl,1],
            alpha=0.6,
            c=colors2[idx],
            marker = markers[idx],
            label="Decision="+str(cl))
    plt.legend(loc='upper left')
    plt.title("Data Distribution with Decision")
    plt.show()
