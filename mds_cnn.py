import numpy as np
import keras
from keras.models import Sequential,Input,Model
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import csv
from time import time

class mds_dl:
	def __init__(self):
		self.train_x_file = ['./database/train_x_20180506.csv','./database/train_x_20180507.csv','./database/train_x_20180508.csv','./database/train_x_test_train.csv']
		self.train_y_file = ['./database/train_y_20180506.csv','./database/train_y_20180507.csv','./database/train_y_20180508.csv','./database/train_y_test_train.csv']
		self.test_file = './database/test_1_x.csv'
		self.p_test = .1				# Test samples percent
		self.p_validation = .15			# Validation samples percent

	def load(self):
		ts=time()
		print('Loading database...')
		data = []
		data_Y = []
		for i in range(len(self.train_x_file)):
			data = np.genfromtxt(self.train_x_file[i], delimiter=',')
			data_Y = np.append(data_Y,np.genfromtxt(self.train_y_file[i], delimiter=','))
			if i==0:
				data_X = data
			else:
				data_X = np.append(data_X, data, axis=0)
			print(data_X.shape)
		data_X = data_X.reshape(-1, 40, 40, 1)
		print(data_X.shape)
		classes = np.unique(data_Y)
		nClasses = len(classes)

		data_Y_one_hot = to_categorical(data_Y)

		train_X, self.test_X, train_label, self.test_label = train_test_split(data_X, data_Y_one_hot, test_size=self.p_test, random_state=42)
		self.train_X, self.valid_X, self.train_label, self.valid_label = train_test_split(train_X, train_label, test_size=self.p_validation, random_state=13)
		print('Training data shape:', self.train_X.shape, self.train_label.shape)
		print('Validating data shape:', self.valid_X.shape, self.valid_label.shape)
		print('Test data shape:', self.test_X.shape, self.test_label.shape)
		print('Total data shape:', data_X.shape, data_Y.shape)
		print('Total number of classes:', nClasses)
		print('Load complete.')
		print('Time cost:', time()-ts,'s')

	def cnn(self):
		ts=time()
		print('Training using CNN...')

		batch_size = 64
		epochs = 10
		num_classes = self.train_label.shape[1]

		model = Sequential()
		model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(40,40,1),padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(MaxPooling2D((2, 2),padding='same'))
		model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=0.1))
		model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
		model.add(LeakyReLU(alpha=0.1))                  
		model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		model.add(Flatten())
		model.add(Dense(128, activation='linear'))
		model.add(LeakyReLU(alpha=0.1))                  
		model.add(Dense(num_classes, activation='softmax'))

		model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

		model.summary()

		model_train = model.fit(self.train_X, self.train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(self.valid_X, self.valid_label))

		test_eval = model.evaluate(self.test_X, self.test_label, verbose=0)
		
		print(model_train.history)
		print('Test loss:', test_eval[0])
		print('Test accuracy:', test_eval[1])
		print('Train complete.')
		print('Time cost:', time()-ts,'s')
		model_json = model.to_json()
		with open('./output/model.json', 'w') as json_file:
			json_file.write(model_json)
		model.save_weights('./output/model.h5')
		print('Saved model to disk.')

	def model_predict(self):
		json_file = open('./output/model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights('./output/model.h5')
		print('Loaded model from disk.')

		test_data = np.genfromtxt(self.test_file, delimiter=',')
		test_data = test_data.reshape(-1,40,40,1)
		print('Test data shape:',test_data.shape)
		# pred = []
		# for i in range(test_data.shape[0]):
			# tmp = test_data[i][:][:][:].reshape(-1,40,40,1)
			# print(tmp.shape)
			# pred = np.append(pred, loaded_model.predict(tmp))
		# pred = pred.reshape(test_data.shape[0],4)
		pred = loaded_model.predict(test_data)
		print(pred.shape)
		np.savetxt('./output/pred.csv', pred, delimiter=',')
		print('Saved.')

	def main(self):
		# self.load()
		# self.cnn()
		self.model_predict()

if __name__ == '__main__':
	mds_dl().main()