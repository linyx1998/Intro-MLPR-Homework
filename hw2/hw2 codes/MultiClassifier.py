import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from ERMClassifier import gaussian_pdf
from ReadData import read_data_3d

def draw_data_3d(predict_labels):
    samples, original_labels = read_data_3d()

    figure = plt.figure()
    axes = Axes3D(figure)
    axes.scatter(samples[0][0], samples[0][1], samples[0][2],\
        color='seagreen',marker='.', alpha=0.6, label='L=1 (True)')
    axes.scatter(samples[0][0], samples[0][1], samples[0][2],\
        color='red',marker='.', alpha=0.6, label='L=1 (False)')
    axes.scatter(samples[0][0], samples[0][1], samples[0][2],\
        color='seagreen',marker='x', alpha=0.6, label='L=2 (True)')
    axes.scatter(samples[0][0], samples[0][1], samples[0][2],\
        color='red',marker='x', alpha=0.6, label='L=2 (False)')
    axes.scatter(samples[0][0], samples[0][1], samples[0][2],\
        color='seagreen',marker='+', alpha=0.6, label='L=3 (True)')
    axes.scatter(samples[0][0], samples[0][1], samples[0][2],\
        color='red',marker='+', alpha=0.6, label='L=3 (False)')

    for i in range(original_labels.size):
        if original_labels[i] == 1 and predict_labels[i] == 1:
            axes.scatter(samples[i][0], samples[i][1], samples[i][2],\
                color='seagreen',marker='.', alpha=0.6)
        elif original_labels[i] == 1 and predict_labels[i] != 1:
            axes.scatter(samples[i][0], samples[i][1], samples[i][2],\
                color='red',marker='.', alpha=0.6)
        elif original_labels[i] == 2 and predict_labels[i] == 2:
            axes.scatter(samples[i][0], samples[i][1], samples[i][2],\
                color='seagreen',marker='x', alpha=0.6)
        elif original_labels[i] == 2 and predict_labels[i] != 2:
            axes.scatter(samples[i][0], samples[i][1], samples[i][2],\
                color='red',marker='x', alpha=0.6)
        elif original_labels[i] == 3 and predict_labels[i] == 3:
            axes.scatter(samples[i][0], samples[i][1], samples[i][2],\
                color='seagreen',marker='+', alpha=0.6)
        else:
            axes.scatter(samples[i][0], samples[i][1], samples[i][2],\
                color='red',marker='+', alpha=0.6)
        
    plt.legend()
    plt.show()

def confusion_matrix():
    cm, cmp = [[0,0,0], [0,0,0], [0,0,0]], [[0,0,0], [0,0,0], [0,0,0]]

    samples, original_labels = read_data_3d()
    predict_labels = map_classify(samples)

    for i in range(predict_labels.size):
        if predict_labels[i] == 1 and original_labels[i] == 1:
            cm[0][0] += 1
        elif predict_labels[i] == 1 and original_labels[i] == 2:
            cm[0][1] += 1
        elif predict_labels[i] == 1 and original_labels[i] == 3:
            cm[0][2] += 1
        elif predict_labels[i] == 2 and original_labels[i] == 1:
            cm[1][0] += 1
        elif predict_labels[i] == 2 and original_labels[i] == 2:
            cm[1][1] += 1
        elif predict_labels[i] == 2 and original_labels[i] == 3:
            cm[1][2] += 1
        elif predict_labels[i] == 3 and original_labels[i] == 1:
            cm[2][0] += 1
        elif predict_labels[i] == 3 and original_labels[i] == 2:
            cm[2][1] += 1
        else:
            cm[2][2] += 1

    cmp[0][0] = cm[0][0]/(cm[0][0]+cm[1][0]+cm[2][0])
    cmp[0][1] = cm[0][1]/(cm[0][1]+cm[1][1]+cm[2][1])
    cmp[0][2] = cm[0][2]/(cm[0][2]+cm[1][2]+cm[2][2])

    cmp[1][0] = cm[1][0]/(cm[0][0]+cm[1][0]+cm[2][0])
    cmp[1][1] = cm[1][1]/(cm[0][1]+cm[1][1]+cm[2][1])
    cmp[1][2] = cm[1][2]/(cm[0][2]+cm[1][2]+cm[2][2])

    cmp[2][0] = cm[2][0]/(cm[0][0]+cm[1][0]+cm[2][0])
    cmp[2][1] = cm[2][1]/(cm[0][1]+cm[1][1]+cm[2][1])
    cmp[2][2] = cm[2][2]/(cm[0][2]+cm[1][2]+cm[2][2])

    print(cmp)
    return(cmp)

def map_classify(samples):
    predict = []
    for i in range(samples.shape[0]):
        predict.append(cal_map(samples[i]))
    return np.array(predict)

def cal_map(sample):
    p1 = 0.3
    p2 = 0.3
    p3 = 0.4

    mean1 = np.matrix([1,1,1]).transpose()
    mean2 = np.matrix([0,0,0]).transpose()
    mean3 = np.matrix([1,0,1]).transpose()
    mean4 = np.matrix([0,1,0]).transpose()
    cov = np.matrix([[2,0,0], [0,2,0], [0,0,2]])

    phi_1 = gaussian_pdf(sample, mean1, cov)*p1
    phi_2 = gaussian_pdf(sample, mean2, cov)*p2
    phi_3 = 0.5*(gaussian_pdf(sample, mean3, cov)+gaussian_pdf(sample, mean4, cov))*p3

    max_phi = max(phi_1, phi_2, phi_3)
    # print(max_phi)

    if max_phi == phi_1:
        return 1
    elif max_phi == phi_2:
        return 2
    else:
        return 3


def eval_erm_classify_loss():
    samples, labels = read_data_3d()
    loss1 = [[0,1,10], [1,0,10], [1,1,0]]
    loss2 = [[0,1,100], [1,0,100], [1,1,0]]

    print("For loss matrix [[0,1,10], [1,0,10], [1,1,0]]:")
    predict_labels, expected_risk = erm_classify_loss(samples, loss1)
    draw_data_3d(predict_labels)
    print("Expected risk: ", expected_risk)

    print("For loss matrix [[0,1,100], [1,0,100], [1,1,0]]:")
    predict_labels, expected_risk = erm_classify_loss(samples, loss2)
    draw_data_3d(predict_labels)
    print("Expected risk: ", expected_risk)

def erm_classify_loss(samples, loss):
    predict = []
    risk = []
    for i in range(samples.shape[0]):
        temp_predict, temp_risk = cal_erm_loss(samples[i], loss)
        predict.append(temp_predict)
        risk.append(temp_risk)
    
    # print(predict)
    # print(np.mean(risk))
    return np.array(predict), np.mean(risk)

def cal_erm_loss(sample, loss):
    p1 = 0.3
    p2 = 0.3
    p3 = 0.4

    mean1 = np.matrix([1,1,1]).transpose()
    mean2 = np.matrix([0,0,0]).transpose()
    mean3 = np.matrix([1,0,1]).transpose()
    mean4 = np.matrix([0,1,0]).transpose()
    cov = np.matrix([[2,0,0], [0,2,0], [0,0,2]])

    px = p1*gaussian_pdf(sample, mean1, cov) + p2*gaussian_pdf(sample, mean2, cov) +\
        p3*0.5*(gaussian_pdf(sample, mean3, cov) + gaussian_pdf(sample, mean4, cov))

    phi_1 = (-1)*(loss[0][0]*gaussian_pdf(sample, mean1, cov)*p1+\
        loss[0][1]*gaussian_pdf(sample, mean2, cov)*p2+\
        loss[0][2]*0.5*(gaussian_pdf(sample, mean3, cov)+gaussian_pdf(sample, mean4, cov))*p3)
    phi_2 = (-1)*(loss[1][0]*gaussian_pdf(sample, mean1, cov)*p1+\
        loss[1][1]*gaussian_pdf(sample, mean2, cov)*p2+\
        loss[1][2]*0.5*(gaussian_pdf(sample, mean3, cov)+gaussian_pdf(sample, mean4, cov))*p3)
    phi_3 = (-1)*(loss[2][0]*gaussian_pdf(sample, mean1, cov)*p1+\
        loss[2][1]*gaussian_pdf(sample, mean2, cov)*p2+\
        loss[2][2]*0.5*(gaussian_pdf(sample, mean3, cov)+gaussian_pdf(sample, mean4, cov))*p3)
    
    max_phi = max(phi_1, phi_2, phi_3)
    # print(max_phi)
    min_risk = (-1)*max_phi/px
    # print(min_risk)

    if max_phi == phi_1:
        return 1, min_risk
    elif max_phi == phi_2:
        return 2, min_risk
    else:
        return 3, min_risk
