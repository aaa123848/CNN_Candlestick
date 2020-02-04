from sklearn.metrics import confusion_matrix
import numpy as np
import argparse

from keras import backend as K
from keras import optimizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras import callbacks as cb
from keras.models import load_model

# customized utilities
from utils import util_process as pro

import matplotlib.pyplot as plt
import mpl_finance as mpf

parser = argparse.ArgumentParser(prog='PROG', description='Tutorial')
parser.add_argument('-i', '-idx', help='choose_the_idx [0~4999]', default=0)
args = parser.parse_args()

def evaluate(input_data, model):
    # load data & keras model
    '''
        Args:
            input_data (numpy) : (1, 32, 32, 4)
            model (h5)
        
        Returns:
            predict result (str)

    '''
    train_pred = model.predict_classes(input_data)

    output_idx = train_pred[0]
    output_result = PATTERN_LS[output_idx]

    return output_result

def result_picture(ts_data, predict_result):
    # show the result picture
    '''
        Args:
            ts_data (numpy) : (1, 32, 4)
            predict_result (str) 

    '''
    plt.close()

    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot2grid((1, 1), (0, 0))
    mpf.candlestick_ohlc(ax, ts_data, width=0.4, alpha=1,
                                colordown='#53c156', colorup='#ff1717')
    plt.title(predict_result)

    plt.show()
    plt.close()


if __name__ == "__main__" :
    PATTERN_LS = ["no_class", "evening", "morning", "bearish_engulfing", "bullish_engulfing",
                    "shooting_star", "inverted_hammer", "bearish_harami", "bullish_harami"]
    PKL_NAME = "./pickle/label8_eurusd_1500_500_val200_gaf_culr.pkl"
    MODEL_PATH= './model/checkpoint_model.h5'

    data = pro.load_pkl(PKL_NAME)
    model = load_model(MODEL_PATH)
    _idx = int(args.i)

    data_gaf = data['test_gaf'][_idx, :, :, :]
    data_gaf = data_gaf.reshape(1, 32, 32, 4)
    ts_data = data['test_data']
    ts_data = np.c_[range(ts_data[_idx, :, :].shape[0]), ts_data[_idx, :, :]]

    predict_result = evaluate(data_gaf, model)
    result_picture(ts_data, predict_result)