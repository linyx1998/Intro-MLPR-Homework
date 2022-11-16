from ReadData import *
from GenerateData import *
from ERMClassifier import *
from EM import *
from ROC import *
from ROC_auto import *
from Regression import *
from vehicle import *

if __name__ == "__main__":
    # generate_data_2d(20000, "Validate_20000")
    # draw_raw_data_2d("Validate_20000")

    # draw_roc_min_error("Validate_20000")
    # draw_min_error("Validate_20000")
    # draw_data_boundary("Validate_20000")
    # class_prior("Train_10000")

    # EM("Train_10000")
    # draw_roc_min_error_auto("Validate_20000", "Train_100")
    # draw_min_error_auto("Validate_20000", "Train_1000")

    # linear_regression("Validate_20000", "Train_100")
    # quadratic_regression("Validate_20000", "Train_1000")

    exec_estimate()

