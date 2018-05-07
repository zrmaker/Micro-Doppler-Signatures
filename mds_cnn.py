import numpy as np
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from time import time

class mds_dl:
	def __init__(self):
		self.train_x_file = './database/train_x.csv'
		self.train_y_file = './database/train_y.csv'
		self.p_test = .1				# Test samples percent
		self.p_validation = .15			# Validation samples percent

	def load(self):
		ts=time()
		print('Loading database...')
		data = np.genfromtxt(self.train_x_file, delimiter=',')
		data_X = data.reshape((data.shape[0],40,40))
		data_X = data_X.reshape(-1, 40, 40, 1)
		data_Y = np.genfromtxt(self.train_y_file, delimiter=',')


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
		epochs = 20
		num_classes = self.train_label.shape[1]

		fashion_model = Sequential()
		fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(40,40,1),padding='same'))
		fashion_model.add(LeakyReLU(alpha=0.1))
		fashion_model.add(MaxPooling2D((2, 2),padding='same'))
		fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
		fashion_model.add(LeakyReLU(alpha=0.1))
		fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
		fashion_model.add(LeakyReLU(alpha=0.1))                  
		fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
		fashion_model.add(Flatten())
		fashion_model.add(Dense(128, activation='linear'))
		fashion_model.add(LeakyReLU(alpha=0.1))                  
		fashion_model.add(Dense(num_classes, activation='softmax'))

		fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

		fashion_model.summary()

		fashion_train = fashion_model.fit(self.train_X, self.train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(self.valid_X, self.valid_label))

		test_eval = fashion_model.evaluate(self.test_X, self.test_label, verbose=0)
		
		print(fashion_train.history)
		print('Test loss:', test_eval[0])
		print('Test accuracy:', test_eval[1])
		print('Time cost:', time()-ts,'s')
		

	def main(self):
		self.load()
		self.cnn()

if __name__ == '__main__':
	mds_dl().main()