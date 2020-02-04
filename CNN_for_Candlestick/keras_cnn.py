from sklearn.metrics import confusion_matrix
import numpy as np

from keras import backend as K
from keras import optimizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras import callbacks as cb

# customized utilities
from utils import util_process as pro


def get_model(params):
    model = Sequential()

    # Conv1
    model.add(Conv2D(20, (4, 4), input_shape=(32, 32, 4), padding='valid', strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    # model.add(Activation('relu'))

    # Conv2
    model.add(Conv2D(20, (4, 4), padding='valid', strides=(1,1)))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    # FC
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))

    model.add(Dense(params['classes']))
    model.add(Activation('softmax'))
    model.summary()
    return model


def train_model(params, data):
    model = get_model(params)
    model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
    # callback objects
    early_stopping = cb.EarlyStopping(monitor='val_acc', patience = params['patience'],
                     restore_best_weights=True, verbose=2)
    saveBestModel = cb.ModelCheckpoint(params['model_path'], monitor='val_acc',
                     save_weights_only=False , save_best_only=True, mode='auto', verbose=1)
    callback_ls = [early_stopping, saveBestModel]
    class_weight = {0: params['zero_weight'], 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}

    print(data.keys())
    print(data['train_label'].shape)
    hist = model.fit(x=data['train_gaf'], y=data['train_label_arr'],
                     validation_data = (data['val_gaf'], data['val_label_arr']),
                     batch_size = params['batch_size'], epochs = params['epochs'],
                     class_weight = class_weight,
                     callbacks = callback_ls, verbose=2)
    return (model, hist)


def print_result(data, model):
    # get train & test pred-labels
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
    PKL_NAME = "./pickle/label8_eurusd_1500_500_val200_gaf_culr.pkl"
    PARAMS = {}
    PARAMS['classes'] = 9
    PARAMS['lr'] = 0.001
    PARAMS['epochs'] = 100
    PARAMS['batch_size'] = 64
    PARAMS['patience'] = 30
    PARAMS['zero_weight'] = 2
    OPTIMIZER = optimizers.SGD(lr=PARAMS['lr'])
    PARAMS['model_path'] = './model/checkpoint_model.h5'
    
    # ---------------------------------------------------------
    # load data & keras model
    data = pro.load_pkl(PKL_NAME)

    # train cnn model
    model, hist = train_model(PARAMS, data)
    # train & test result
    print_result(data, model)

