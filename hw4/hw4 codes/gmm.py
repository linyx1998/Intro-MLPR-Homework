import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

def kfold(X):
    likelihood_select_train = []
    likelihood_select_test = []
    aic_select_train = []
    aic_select_test = []
    bic_select_train = []
    bic_select_test = []

    print(X.shape)
    for i in range(100):
        avg_log_likelihood_train = []
        avg_log_likelihood_test = []
        avg_aic_train = []
        avg_aic_test = []
        avg_bic_train = []
        avg_bic_test = []

        for components in range(1, 7):
            cv = KFold(n_splits=10, shuffle=True)
            log_train = 0
            log_test = 0
            aic_train = 0
            aic_test = 0
            bic_train = 0
            bic_test = 0

            for train_index, test_index in cv.split(X):
                X_train, X_test= X[train_index], X[test_index]
                # print(X_train.shape,X_test.shape)
                train, test, atrain, atest, btrain, btest = \
                    log_likelihood(X_train, X_test, components)
                log_train += train
                log_test += test
                aic_train += atrain
                aic_test += atest
                bic_train += btrain
                bic_test += btest

            avg_log_likelihood_train.append((float)(log_train/10))
            avg_log_likelihood_test.append((float)(log_test/10))
            avg_aic_train.append(aic_train/10)
            avg_aic_test.append(aic_test/10)
            avg_bic_train.append(bic_train/10)
            avg_bic_test.append(bic_test/10)
        
        likelihood_select_train.append(avg_log_likelihood_train.index(max(avg_log_likelihood_train))+1)
        likelihood_select_test.append(avg_log_likelihood_test.index(max(avg_log_likelihood_test))+1)
        aic_select_train.append(avg_aic_train.index(min(avg_aic_train))+1)
        aic_select_test.append(avg_aic_test.index(min(avg_aic_test))+1)
        bic_select_train.append(avg_bic_train.index(min(avg_bic_train))+1)
        bic_select_test.append(avg_bic_test.index(min(avg_bic_test))+1)

        # draw_scores(avg_log_likelihood_train, avg_aic_train, avg_bic_train, \
            # avg_log_likelihood_test, avg_aic_test, avg_bic_test, X.shape[0])

    # print(likelihood_select_train, likelihood_select_test)
    print(likelihood_select_train.count(1), likelihood_select_train.count(2), likelihood_select_train.count(3),
        likelihood_select_train.count(4), likelihood_select_train.count(5), likelihood_select_train.count(6))
    print(likelihood_select_test.count(1), likelihood_select_test.count(2), likelihood_select_test.count(3),
        likelihood_select_test.count(4), likelihood_select_test.count(5), likelihood_select_test.count(6))
    print(aic_select_train.count(1), aic_select_train.count(2), aic_select_train.count(3),
        aic_select_train.count(4), aic_select_train.count(5), aic_select_train.count(6))
    print(aic_select_test.count(1), aic_select_test.count(2), aic_select_test.count(3),
        aic_select_test.count(4), aic_select_test.count(5), aic_select_test.count(6))
    print(bic_select_train.count(1), bic_select_train.count(2), bic_select_train.count(3),
        bic_select_train.count(4), bic_select_train.count(5), bic_select_train.count(6))
    print(bic_select_test.count(1), bic_select_test.count(2), bic_select_test.count(3),
        bic_select_test.count(4), bic_select_test.count(5), bic_select_test.count(6))



def draw_scores(train_likelihood, train_aic, train_bic, test_likelihood, test_aic, test_bic, N):
    x = [1,2,3,4,5,6]
    colors = ('darkorange','skyblue','lightcoral','mediumpurple')

    fig, ax = plt.subplots(2,3)
    ax[0,0].plot(x, train_likelihood, c=colors[0], marker='x')
    ax[0,0].set_title('average log-likelihoods (Training)')

    ax[0,1].plot(x, train_aic, c=colors[1], marker='x')
    ax[0,1].set_title('aic scores (Training)')

    ax[0,2].plot(x, train_bic, c=colors[2], marker='x')
    ax[0,2].set_title('bic scores (Training)')

    ax[1,0].plot(x, test_likelihood, c=colors[0], marker='x')
    ax[1,0].set_title('average log-likelihoods (Testing)')

    ax[1,1].plot(x, test_aic, c=colors[1], marker='x')
    ax[1,1].set_title('aic scores (Testing)')

    ax[1,2].plot(x, test_bic, c=colors[2], marker='x')
    ax[1,2].set_title('bic scores (Testing)')

    # plt.suptitle('N='+str(N))
    plt.show()



def log_likelihood(X_train, X_test, components):
    clst = mixture.GaussianMixture(n_components=components)
    clst.fit(X_train)

    # x = np.linspace(-15., 15.)
    # y = np.linspace(-15., 15.)
    # X, Y = np.meshgrid(x, y)
    # XX = np.array([X.ravel(), Y.ravel()]).T
    # Z = -clst.score_samples(XX)
    # Z = Z.reshape(X.shape)

    # CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=100.0),
    #                 levels=np.logspace(0, 1, 10))
    # # CB = plt.colorbar(CS, shrink=0.8, extend='both')
    # plt.scatter(X_train[:, 0], X_train[:, 1])

    # plt.title('Data Distribution')
    # plt.axis('tight')
    # plt.show()

    # print(clst.score(samples))
    return clst.score(X_train), clst.score(X_test),\
        clst.aic(X_train), clst.aic(X_test), clst.bic(X_train), clst.bic(X_test)
