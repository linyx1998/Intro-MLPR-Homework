from ReadData import *
from GenerateData import *
from ERMClassifier import *
from LDAClassifier import *
from MultiClassifier import *
from ROC import *

if __name__ == "__main__":
    # Question1
    # generate_data_2d(10000)
    # draw_raw_data_2d()

    # Qustion1 PartA Q2
    # draw_roc()

    # Qustion1 PartA Q3
    # draw_roc_min_error()

    # Question1 PartB
    # draw_lda_roc()

    # Question2 PartA Q1
    # generate_data_3d(10000)
    # draw_raw_data_3d()

    # Question2 PartA Q2
    # confusion_matrix()

    # Question2 PartA Q3
    # draw_data_3d(map_classify(read_data_3d()[0]))

    # Question2 PartB
    eval_erm_classify_loss()
