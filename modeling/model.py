import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, Activation, Dropout, BatchNormalization, ActivityRegularization, \
    Embedding, Flatten
import os
from keras.callbacks import ModelCheckpoint, EarlyStopping
import json


class feedforwardnn:
    '''
    Deep feedforward neural network object, model built using information from the configurations json file
    All layers are fully-connected layers, with regularisation, batch normalisation and dropouts for variance
    reduction purposes.
    '''
    def __init__(self):
        self.nn = keras.Sequential()

    def load_model(self, fpath):
        print("Loading pre-trained model from %s" % fpath)
        self.nn = load_model(fpath)

    def build_model(self, configs):
        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            embedding_input_size = layer['input'] if 'embedding' in layer else None
            embedding_output_size = layer['output'] if 'embedding' in layer else None
            l2_penalisation = layer['l2penalty']

            if layer['type'] == 'Embedding':
                self.nn.add(Embedding(embedding_input_size, embedding_output_size))

            if layer['type'] == 'Flatten':
                self.nn.add(Flatten())

            if layer['type'] == 'Dense':
                self.nn.add(Dense(neurons, activation=activation))

            if layer['type'] == 'Normalisation':
                self.nn.add(BatchNormalization)

            if layer['type'] == 'Regularisation':
                self.nn.add(ActivityRegularization(l2=l2_penalisation))
                
            if layer['type'] == 'Dropout':
                self.nn.add(Dropout(neurons, dropout_rate=dropout_rate))

        self.nn.compile(optimizer=configs['model']['optimiser'], loss=configs['model']['loss'])

    def train(self, x, y, epochs, batch_size, sav_dir):
        sav_filename = os.path.join(sav_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
        callbacks = [EarlyStopping(patience=2),
                    ModelCheckpoint(filepath=sav_filename, save_best_only=True)]
        self.nn.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        self.nn.save(sav_filename)

    def predict(self, data):
        prediction = self.nn.predict(data)
        prediction = np.reshape(prediction, (prediction.size,))
        return prediction


def plot_predictions(predictions, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True data')
    ax.plot(predictions, label='Prediction')
    plt.legend()
    plt.show()


def main():
    configs = json.load(open('configs.json', 'r'))
    if not os.path.exists(configs['model']['sav_dir']): os.mkdir(configs['model']['sav_dir'])

    # here we handle the data into x, y and other sort of things
    nn = feedforwardnn()
    nn.build_model(configs)

    # steps_per_epoch = math.ceil((train_len - configs['data']['sequence_length']) / configs['training']['batch_size'])
    nn.train(x, y, epochs=configs['model']['epochs'], batch_size=configs['model']['batch_size'],
             sav_dir=configs['model']['sav_dir'])
    x_test, y_test = ..., ...
    predictions = lstm.predict_point(x_test)
    plot_predictions(predictions, y_test)


if __name__ == '__main__':
    main()











