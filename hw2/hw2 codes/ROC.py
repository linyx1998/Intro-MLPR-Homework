from ReadData import *
from GenerateData import *
from ERMClassifier import *
from LDAClassifier import *
import matplotlib.pyplot as plt
import numpy as np

def draw_roc():
    samples = read_data_2d()[0]
    labels = read_data_2d()[1]
    # samples, labels = generate_data_2d(10000)

    x, y, bg, bx, by, tg, tx, ty = estimate_roc(samples, labels)
    plt.scatter(x, y)
    plt.title("Approximation of the ROC Curve")
    plt.xlabel("P(D = 1|L = 0; gamma)")
    plt.ylabel("P(D = 1|L = 1; gamma)")

    plt.show()
    return 0

def draw_roc_min_error():
    samples = read_data_2d()[0]
    labels = read_data_2d()[1]
    # samples, labels = generate_data_2d(10000)

    x, y, bg, bx, by, tg, tx, ty = estimate_roc(samples, labels)
    plt.scatter(x, y)

    # bx, by is determined empirically to minimize p(error)
    best_gamma = str(bg).split('.')[0]+'.'+str(bg).split('.')[1][:3]
    plt.scatter(bx, by, color='hotpink', marker='*', s=80,\
        label='gamma='+best_gamma+' (empirically optimal)')
    
    # tx, ty is determined theoretically to minimize p(error)
    theoretical_gamma = str(tg).split('.')[0]+'.'+str(tg).split('.')[1][:3]
    plt.scatter(tx, ty, color='yellowgreen', marker='x', s=80,\
        label='gamma='+theoretical_gamma+' (theoretically optimal)')
    
    plt.title("Approximation of the ROC Curve")
    plt.xlabel("P(D = 1|L = 0; gamma)")
    plt.ylabel("P(D = 1|L = 1; gamma)")
    
    plt.legend()
    plt.show()
    return 0

def estimate_roc(samples, labels):
    gamma_1 = np.append(np.arange(0, 0.2, 0.01), np.arange(0.1, 2, 0.1))
    gamma_2 = np.append(np.arange(1, 20, 1), np.arange(10, 200, 10))
    gamma_3 = np.append(np.arange(100, 2000, 100), np.arange(1000, 10000, 1000))
    gamma = np.concatenate((gamma_1, np.arange(1.8, 1.9, 0.001), gamma_2, gamma_3))

    original_label = labels
    scatter_x = []
    scatter_y = []

    min_error = 1.0
    best_gamma = gamma[0]
    best_x = 0
    best_y = 0

    for i in range(gamma.size):
        predict_label = erm_classify(samples, gamma[i])
        tp, fp, fn, tn = 0, 0, 0, 0
        for j in range(predict_label.size):
            if predict_label[j] == original_label[j] and predict_label[j] == 1:
                tp += 1
            elif predict_label[j] == original_label[j] and predict_label[j] == 0:
                tn += 1
            elif predict_label[j] != original_label[j] and predict_label[j] == 1:
                fp += 1
            else:
                fn += 1
        err = (fp+fn)/predict_label.size
        if err < min_error:
            min_error = err
            best_gamma = gamma[i]
            best_x = fp/(fp+tn)
            best_y = tp/(tp+fn)
        
        scatter_x.append(fp/(fp+tn))
        scatter_y.append(tp/(tp+fn))
    
    theo_gamma = 0.65/0.35
    predict_label = erm_classify(samples, theo_gamma)
    tp, fp, fn, tn = 0, 0, 0, 0
    for j in range(predict_label.size):
        if predict_label[j] == original_label[j] and predict_label[j] == 1:
            tp += 1
        elif predict_label[j] == original_label[j] and predict_label[j] == 0:
            tn += 1
        elif predict_label[j] != original_label[j] and predict_label[j] == 1:
            fp += 1
        else:
            fn += 1
    theo_error = (fp+fn)/predict_label.size
    theo_x = fp/(fp+tn)
    theo_y = tp/(tp+fn)

    print("min error theoretically:", theo_error)
    print("min error empirically:", min_error)
    
    return np.array(scatter_x), np.array(scatter_y),\
        best_gamma, best_x, best_y, theo_gamma, theo_x, theo_y


def draw_lda_roc():
    samples = read_data_2d()[0]
    labels = read_data_2d()[1]

    x, y, bt, bx, by = estimate_lda_roc(samples, labels)
    plt.scatter(x, y)

    best_tau = str(bt).split('.')[0]+'.'+str(bt).split('.')[1][:3]
    plt.scatter(bx, by, color='hotpink', marker='*', s=80,\
        label='tau='+best_tau+' (empirically optimal)')

    plt.title("Approximation of the ROC Curve (LDA)")
    plt.xlabel("P(D = 1|L = 0; gamma)")
    plt.ylabel("P(D = 1|L = 1; gamma)")

    plt.legend()
    plt.show()
    return 0

def estimate_lda_roc(samples, labels):
    tau = np.append(np.arange(-10, 10, 0.1), np.arange(-1, 1, 0.01))
    tau = np.concatenate((tau, np.arange(-5, -4, 0.001)))

    original_label = labels
    scatter_x = []
    scatter_y = []

    min_error = 1.0
    best_tau = tau[0]
    best_x = 0
    best_y = 0

    for i in range(tau.size):
        predict_label = lda_classify(samples, labels, tau[i])
        tp, fp, fn, tn = 0, 0, 0, 0
        for j in range(predict_label.size):
            if predict_label[j] == original_label[j] and predict_label[j] == 1:
                tp += 1
            elif predict_label[j] == original_label[j] and predict_label[j] == 0:
                tn += 1
            elif predict_label[j] != original_label[j] and predict_label[j] == 1:
                fp += 1
            else:
                fn += 1
        err = (fp+fn)/predict_label.size
        if err < min_error:
            min_error = err
            best_tau = tau[i]
            best_x = fp/(fp+tn)
            best_y = tp/(tp+fn)
        
        scatter_x.append(fp/(fp+tn))
        scatter_y.append(tp/(tp+fn))

    print("min error empirically:", min_error)
    
    return np.array(scatter_x), np.array(scatter_y), best_tau, best_x, best_y