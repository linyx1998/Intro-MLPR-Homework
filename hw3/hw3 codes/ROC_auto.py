from ReadData import *
from GenerateData import *
from ERMClassifier import *
import matplotlib.pyplot as plt
import numpy as np
from EM import *

def draw_roc_min_error_auto(name, data_name):
    samples = read_data_2d(name)[0]
    labels = read_data_2d(name)[1]

    x, y, bg, bx, by, tg, tx, ty = estimate_roc_auto(samples, labels, data_name)

    plt.plot(x, y)

    # plt.scatter(x,y)

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

def draw_min_error_auto(name, data_name):
    samples = read_data_2d(name)[0]
    labels = read_data_2d(name)[1]

    x, y, bg, tg, te, me = estimate_error_auto(samples, labels, data_name)
    plt.plot(y[:500], x[:500])

    # bx, by is determined empirically to minimize p(error)
    best_gamma = str(bg).split('.')[0]+'.'+str(bg).split('.')[1][:3]
    plt.scatter(bg, me, color='hotpink', marker='*', s=80,\
        label='gamma='+best_gamma+' (empirically optimal)')
    
    # tx, ty is determined theoretically to minimize p(error)
    theoretical_gamma = str(tg).split('.')[0]+'.'+str(tg).split('.')[1][:3]
    plt.scatter(tg, te, color='yellowgreen', marker='x', s=80,\
        label='gamma='+theoretical_gamma+' (theoretically optimal)')
    
    plt.title("Relationship Between P(error) and Thresholds")
    plt.ylabel("P(error)")
    plt.xlabel("Thresholds")
    
    plt.legend()
    plt.show()
    return 0

def estimate_roc_auto(samples, labels, data_name):
    gamma_1 = np.append(np.arange(0, 0.2, 0.01), np.arange(0.2, 2, 0.1))
    gamma_2 = np.append(np.arange(2, 20, 1), np.arange(20, 200, 10))
    gamma_3 = np.append(np.arange(200, 2000, 100), np.arange(2000, 10000, 1000))
    gamma = np.concatenate((gamma_1, gamma_2, gamma_3, np.arange(1.46, 1.54, 0.001)))
    gamma.sort()

    original_label = labels
    scatter_x = []
    scatter_y = []
    p_error = []

    min_error = 1.0
    best_gamma = gamma[0]
    best_x = 0
    best_y = 0

    for i in range(gamma.size):
        predict_label = erm_classify_auto(samples, gamma[i], data_name)
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
        p_error.append(err)
        scatter_x.append(fp/(fp+tn))
        scatter_y.append(tp/(tp+fn))
    
    cp1, cp0 = class_prior(data_name)
    theo_gamma = cp0/cp1
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

    scatter_x, scatter_y = zip(*sorted(zip(scatter_x, scatter_y)))
    
    return np.array(scatter_x), np.array(scatter_y),\
        best_gamma, best_x, best_y, theo_gamma, theo_x, theo_y

def estimate_error_auto(samples, labels, data_name):
    gamma_1 = np.append(np.arange(0, 0.2, 0.01), np.arange(0.2, 2, 0.1))
    gamma_2 = np.append(np.arange(2, 20, 1), np.arange(20, 200, 10))
    gamma = np.concatenate((gamma_1, gamma_2, np.arange(1.47, 1.53, 0.001)))
    gamma.sort()

    original_label = labels
    scatter_x = []
    scatter_y = []
    p_error = []

    min_error = 1.0
    best_gamma = gamma[0]

    for i in range(gamma.size):
        predict_label = erm_classify_auto(samples, gamma[i], data_name)
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
        p_error.append(err)
        scatter_x.append(fp/(fp+tn))
        scatter_y.append(tp/(tp+fn))
    
    cp1, cp0 = class_prior(data_name)
    theo_gamma = cp0/cp1
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

    print("min error theoretically:", theo_error)
    print("min error empirically:", min_error)
    
    return np.array(p_error), np.array(gamma), best_gamma, theo_gamma, theo_error, min_error
