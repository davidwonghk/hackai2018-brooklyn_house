import os
import json

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv1D, Activation, Dropout, BatchNormalization, ActivityRegularization, Embedding, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split


class feedforwardnn:
	'''
	Deep feedforward neural network object, model built using information from the configurations json file
	All layers are fully-connected layers, with regularisation, batch normalisation and dropouts for variance
	reduction purposes.
	'''
	def __init__(self):
		self.model = Sequential()


	def build_model(self, configs):
		for layer in configs['model']['layers']:
			if layer['type'] == 'Embedding':
				embedding_input_size = layer['input_size']
				embedding_output_size = layer['output_size']
				self.model.add(Embedding(embedding_input_size, embedding_output_size))

			#if layer['type'] == 'Flatten':
				#self.model.add(Flatten())

			if layer['type'] == 'Dense':
				neurons = layer['neurons']
				activation = layer['activation']
				if 'input_size' in layer:
					input_size = layer['input_size']
					self.model.add(Dense(neurons, activation=activation, input_shape=(input_size,)))
				else:
					self.model.add(Dense(neurons, activation=activation))

			if layer['type'] == 'Normalisation':
				self.model.add(BatchNormalization())

			if layer['type'] == 'Regularisation':
				l2_penalisation = layer['l2_penalisation']
				self.model.add(ActivityRegularization(l2=l2_penalisation))

			if layer['type'] == 'Dropout':
				dropout_rate = layer['rate']
				self.model.add(Dropout(dropout_rate))

			self.model.compile(optimizer=configs['model']['optimiser'], loss=configs['model']['loss'])

	def train(self, x, y, epochs, batch_size, sav_dir):
		sav_filename = os.path.join(sav_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [EarlyStopping(patience=2),
		ModelCheckpoint(filepath=sav_filename, save_best_only=True)]
		self.model.fit(x, y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
		self.model.save(sav_filename)

	def predict(self, data):
		prediction = self.model.predict(data)
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

	#df = pd.read_csv("data/pre_processed.csv")
	X, y = np.split(df,[-1],axis=1)
	X,y, _, _ =  train_test_split(X, y, test_size=0.1, random_state=1)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

	# steps_per_epoch = math.ceil((train_len - configs['data']['sequence_length']) / configs['training']['batch_size'])
	nn.train(X_train, y_train, epochs=configs['model']['epochs'], batch_size=configs['model']['batch_size'],
	sav_dir=configs['model']['sav_dir'])

	predictions = lstm.predict_point(x_test)
	plot_predictions(predictions, y_test)


if __name__ == '__main__':
	main()
