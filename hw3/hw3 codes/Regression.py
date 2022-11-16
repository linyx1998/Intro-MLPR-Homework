from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from ReadData import read_data_2d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay

def linear_regression(name, data_name):
    samples, labels = read_data_2d(data_name)
    linreg = LinearRegression()
    linreg.fit(samples, labels)

    print(linreg.intercept_)
    print(linreg.coef_)

    test_samples, test_labels = read_data_2d(name)
    predict_label = linreg.predict(test_samples)
    print(predict_label)
    for j in range(predict_label.size):
        if predict_label[j]>=0.5:
            predict_label[j]=1
        else:
            predict_label[j]=0

    tp, fp, fn, tn = 0, 0, 0, 0
    for j in range(predict_label.size):
        if predict_label[j] == test_labels[j] and predict_label[j] == 1:
            tp += 1
        elif predict_label[j] == test_labels[j] and predict_label[j] == 0:
            tn += 1
        elif predict_label[j] != test_labels[j] and predict_label[j] == 1:
            fp += 1
        else:
            fn += 1
    print(fn+fp, predict_label.size)
    err = (fp+fn)/predict_label.size
    print("P(error):", err)

    markers = ('+','x')
    colors = ('darkorange','skyblue')
    colors2 = ('orangered', 'lime')
    cmap = ListedColormap(colors[:len(np.unique(test_labels))])

    x1_min,x1_max = test_samples[:,0].min()-1,test_samples[:,0].max()+1
    x2_min,x2_max = test_samples[:,1].min()-1,test_samples[:,1].max()+1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,0.05),
                         np.arange(x2_min,x2_max,0.05))  
    Z = linreg.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    for i in range(Z.size):
        if Z[i]<=0.5:
            Z[i] = 0
        else:
            Z[i] = 1
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    decision = []
    for i in range(test_labels.size):
        if test_labels[i]==predict_label[i]:
            decision.append(1)
        else:
            decision.append(0)

    decision = []
    for i in range(test_labels.size):
        decision.append(test_labels[i]==predict_label[i])
    decision = np.array(decision)

    # plot class samples
    for idx,cl in enumerate(np.unique(test_labels)):
        plt.scatter(x=test_samples[test_labels==cl,0],
            y = test_samples[test_labels==cl,1],
            alpha=0.6,
            c=colors[idx],
            marker = markers[idx],
            label="L="+str(cl))
    # for idx,cl in enumerate(np.unique(decision)):
    #     plt.scatter(x=test_samples[decision==cl,0],
    #         y = test_samples[decision==cl,1],
    #         alpha=0.6,
    #         c=colors2[idx],
    #         marker = markers[idx],
    #         label="Decision="+str(cl))
    plt.legend(loc='upper left')
    plt.title("Data Distribution with Decision ("+data_name+')')
    plt.show()

def quadratic_regression(name, data_name):
    samples, labels = read_data_2d(data_name)

    polyreg = Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("std_scaler", StandardScaler()),
        ("lin_reg", LinearRegression())
    ])
    polyreg.fit(samples, labels)

    test_samples, test_labels = read_data_2d(name)

    predict_label = polyreg.predict(test_samples)
    print(predict_label)

    for j in range(predict_label.size):
        if predict_label[j]>=0.5:
            predict_label[j]=1
        else:
            predict_label[j]=0

    tp, fp, fn, tn = 0, 0, 0, 0
    for j in range(predict_label.size):
        if predict_label[j] == test_labels[j] and predict_label[j] == 1:
            tp += 1
        elif predict_label[j] == test_labels[j] and predict_label[j] == 0:
            tn += 1
        elif predict_label[j] != test_labels[j] and predict_label[j] == 1:
            fp += 1
        else:
            fn += 1
    print(fn+fp, predict_label.size)
    err = (fp+fn)/predict_label.size
    print("P(error):", err)

    markers = ('+','x')
    colors = ('darkorange','skyblue')
    colors2 = ('orangered', 'lime')
    cmap = ListedColormap(colors[:len(np.unique(test_labels))])

    x1_min,x1_max = test_samples[:,0].min()-1,test_samples[:,0].max()+1
    x2_min,x2_max = test_samples[:,1].min()-1,test_samples[:,1].max()+1
    xx1,xx2 = np.meshgrid(
        # np.arange(x1_min,x1_max,0.05),
        # np.arange(x2_min,x2_max,0.05)
        np.linspace(x1_min, x1_max, int((x1_max-x1_min)*100)).reshape(-1, 1),
        np.linspace(x2_min, x2_max, int((x2_max-x2_min)*100)).reshape(-1, 1),
        )  
    # Z = polyreg.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    # Z = Z.reshape(xx1.shape)
    Z = polyreg.predict(np.c_[xx1.ravel(), xx2.ravel()])
    for i in range(Z.size):
        if Z[i]<=0.5:
            Z[i] = 0
        else:
            Z[i] = 1
    print(Z)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    decision = []
    for i in range(test_labels.size):
        decision.append(test_labels[i]==predict_label[i])
    decision = np.array(decision)

    # plot class samples
    for idx,cl in enumerate(np.unique(test_labels)):
        plt.scatter(x=test_samples[test_labels==cl,0],
            y = test_samples[test_labels==cl,1],
            alpha=0.6,
            c=colors[idx],
            marker = markers[idx],
            label="L="+str(cl))

    # for idx,cl in enumerate(np.unique(decision)):
    #     plt.scatter(x=test_samples[decision==cl,0],
    #         y = test_samples[decision==cl,1],
    #         alpha=0.6,
    #         c=colors2[idx],
    #         marker = markers[idx],
    #         label="Decision="+str(cl))
    plt.legend(loc='upper left')
    plt.title("Data Distribution with Decision Boundary (trained by:"+data_name+')')
    plt.show()