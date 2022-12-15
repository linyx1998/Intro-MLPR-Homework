from GenerateData import *
from ReadData import *
from classify import *
from gmm import *

if __name__ == "__main__":
    N_train = 5000
    N_test = 100000

    # generate_data(N_train)
    # draw_raw_data(N_train)

    X_train, y_train = read_data(N_train)
    X_test, y_test = read_data(N_test)

    optimal_classifier(X_train, y_train)

    # model_selection(X_train, y_train)
    # draw_model_selection()
    
    # optimal_classifier(X_test, y_test)

    # mlp_classifier(X_train, y_train, X_test, y_test, 30)

    # draw_test_error()

    # N = 10000
    # generate_data_2(N)
    # draw_raw_data_2(N)

    # X, y = read_data_2(N)
    # kfold(X)
