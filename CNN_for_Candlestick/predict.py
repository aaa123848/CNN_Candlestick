from sklearn.metrics import confusion_matrix
import numpy as np

from keras import backend as K
from keras import optimizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras import callbacks as cb
from keras.models import load_model

# customized utilities
from utils import util_process as pro



def evaluate(data, model_path):
    # get train & test pred-labels
    model = load_model(model_path)
    train_pred = model.predict_classes(data['train_gaf'])
    test_pred = model.predict_classes(data['test_gaf'])
    # get train & test true-labels
    train_label = data['train_label'][:, 0]
    test_label = data['test_label'][:, 0]
    # train & test confusion matrix
    train_result_cm = confusion_matrix(train_label, train_pred, labels=range(9))
    test_result_cm = confusion_matrix(test_label, test_pred, labels=range(9))
    print(train_result_cm, '\n')
    print(test_result_cm, '\n')
    print(train_result_cm[0, 0] / np.sum(train_result_cm[0, :]),
          test_result_cm[0, 0] / np.sum(test_result_cm[0, :]))


if __name__ == "__main__":
    PARAMS = {}
    PKL_NAME = "./pickle/label8_eurusd_1500_500_val200_gaf_culr.pkl"

    PARAMS['model_path'] = './model/checkpoint_model.h5'
    
    # ---------------------------------------------------------
    # load data & keras model
    data = pro.load_pkl(PKL_NAME)
    model_path = PARAMS['model_path']

    evaluate(data, model_path)

