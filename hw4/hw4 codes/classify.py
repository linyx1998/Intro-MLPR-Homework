from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
from ReadData import *

def mlp_classifier(X_train, y_train, X_test, y_test, P):
    clf = MLPClassifier(activation = 'tanh', learning_rate_init = 0.001, 
        hidden_layer_sizes=(P,), max_iter = 1000).fit(X_train, y_train)
    print("train acc:", np.mean(clf.predict(X_train) == y_train))
    print("test acc:", np.mean(clf.predict(X_test) == y_test))
    print("p error:", 1-np.mean(clf.predict(X_test) == y_test))

def map_classifier(X_train, y_train):
    gnb = GaussianNB().fit(X_train, y_train)
    # print(gnb.theta_)
    print(1.0-np.mean(gnb.predict(X_train) == y_train))

def optimal_classifier(X, y):
    mean1 = (-5,3,-2)
    mean2 = (2,5,3)
    mean3 = (5,-2,-2)
    mean4 = (0,-5,-3)

    cov1 = [[10,3,-2], [3,5,0], [-2,0,10]]
    cov2 = [[5,1,0], [1,10,0], [0,0,5]]
    cov3 = [[10,-2,0], [-2,5,0], [0,0,10]]
    cov4 = [[10,0,0], [0,10,5], [0,5,10]]

    pdf0 = multivariate_normal.pdf(X, mean1, cov1)
    pdf1 = multivariate_normal.pdf(X, mean2, cov2)
    pdf2 = multivariate_normal.pdf(X, mean3, cov3)
    pdf3 = multivariate_normal.pdf(X, mean4, cov4)

    wrong_decision = 0
    for i in range(y.shape[0]):
        pdfs = np.array([pdf0[i], pdf1[i], pdf2[i], pdf3[i]])
        if y[i] != (np.argmax(pdfs)+1):
            wrong_decision += 1
    print(wrong_decision/y.shape[0])


def model_selection(X, y):
    avg_error = []
    for p in range(1, 31):
        clf = MLPClassifier(activation = 'tanh', learning_rate_init = 0.001, 
            hidden_layer_sizes=(p,), max_iter = 2000)
        scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
        # print(scores)
        # print(scores.mean())
        avg_error.append((1.0-scores.mean()))
    print(avg_error)
    return avg_error

def draw_model_selection():
    X = []
    y = []
    N_list = [100, 200, 500, 1000, 2000, 5000]
    for i in range(len(N_list)):
        temp_X, temp_y = read_data(N_list[i])
        X.append(temp_X)
        y.append(temp_y)

    avg_errors = []
    for i in range(len(N_list)):
        avg_error = model_selection(X[i], y[i])
        avg_errors.append(avg_error)
    p_list = []
    for i in range(len(avg_errors[0])):
        p_list.append(i+1)

    best_p = []
    best_error = []
    for i in range(len(N_list)):
        best_error.append(min(avg_errors[i]))
        best_p.append(avg_errors[i].index(min(avg_errors[i]))+1)
    
    print()
    print('best_error',best_error)
    print('best_p',best_p)

    colors = ('darkorange','skyblue','lightcoral','mediumpurple',
        'darkcyan','chocolate')
    for i in range(len(N_list)):
        plt.plot(p_list, avg_errors[i], label='N='+str(N_list[i]), c=colors[i])
    plt.title("Model Order Selection")
    plt.legend()
    plt.show()

def draw_test_error():
    x = [100,200,500,1000,2000,5000]
    y = [0.167,0.153,0.140,0.134,0.132,0.129]
    optim = 0.128

    plt.plot(x, y, label="MLP P(error)", marker='^', c='darkcyan')
    plt.hlines(optim,x[0],x[-1],linestyles='dashed', label="Optimal P(error)")
    plt.title("P(error) on Test Dataset")
    plt.legend()
    plt.show()
