import numpy as np 
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D
from keras.optimizers import RMSprop, SGD, Adam
from keras.wrappers.scikit_learn import KerasRegressor
import random
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
from sklearn.metrics import make_scorer
import scipy.stats as stats
import datetime, time

np.random.seed(1337)


#hyperparams include:
#	optimizer ['adam','rms']
#	loss ['mean_squared_error','mean_squared_logarithmic']
#	learning rate [0.001, 0.01, 0.1]
#	hidden layer neurons [(seq_len/8),(seq_len/4),(seq_len/2)]
#	filter batches [20,40,60]
#	filter lengths [(seq_len/50),(seq_len/25),(seq_len/12),(seq_len/6),(seq_len/3)]
#	batch size [32,64,128]
#	epochs [5,10,15]
def create_model(learn_rate=0.001, neurons = 1, seq_len=15):
	"""Builds a parameterized CNN Model"""
	cnn = Sequential()
	cnn.add(Convolution1D(nb_filter=30,filter_length=6,input_dim=4,input_length=seq_len,border_mode="same", activation='relu'))
	cnn.add(Dropout(0.1))
	cnn.add(Convolution1D(nb_filter=40,filter_length=6,input_dim=4,input_length=seq_len,border_mode="same", activation='relu'))

	cnn.add(Flatten())

	cnn.add(Dense(neurons))
	cnn.add(Dropout(0.2))
	cnn.add(Activation('relu'))

	cnn.add(Dense(1))
	cnn.add(Activation('linear'))

	#compile the model
	adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	rms = RMSprop(lr=learn_rate, rho=0.9, epsilon=1e-08)

	cnn.compile(loss='mean_squared_error', optimizer=adam)

	return cnn

def my_custom_r2_func(ground_truth, predictions):
	slope, intercept, r_value, p_value, std_err = stats.linregress(ground_truth.reshape(-1),predictions.reshape(-1))
	print "..."
	return r_value**2
	




class dnaModel(object):
	""" An optimal CNN model for DNA sequences """

	def __init__(self, df, filename = ''):
		"""Initialize dnaModel object
		The input df should contain only two columns: sequence + expression"""

		self.filename = filename
		self.df = df
		self.seq_len, self.X_train, self.Y_train, self.X_test, self.Y_test = self.__parse_input()
		self.model = Sequential()
		self.predicted = []
		
	def __parse_input(self):
		""" Splits training and test data from the input dataframe """
		xtrain, ytrain, xtest, ytest = [], [], [], []

		df = self.df
		############# Format NN inputs #############
		seq_len = len(df['sequence'][0])
		X_data = np.empty([len(df),seq_len,4])
		indx = 0

		Y_data = np.array(df[['output1']])

		for seq in df['sequence']:
			X_data[indx] = self.__oneHotEncoder(seq)
			indx += 1


		########## RANDOM TEST/TRAIN SPLIT #########
		normed_out = preprocessing.StandardScaler().fit_transform(Y_data)
		xtrain, xtest, ytrain, ytest = train_test_split(X_data, normed_out, test_size=0.15, random_state=42)

		return seq_len, xtrain, ytrain, xtest, ytest

	def __opt_model(self):
		"""bulids a new model and optimizes hyperparams
		trains/fits and saves best model and params. 
		Currently a simple gridsearch, but should switch to a stochaistic 
		optimization alg with mean performance as the cost fcn """
		
		seq_len, xtrain, ytrain, xtest, ytest = self.seq_len, self.X_train, self.Y_train, self.X_test, self.Y_test

		model = KerasRegressor(build_fn=create_model, nb_epoch=6, batch_size=128, verbose=0)

		# define the grid search parameters
		num_bases = [seq_len]
		learn_rate = [0.001]
		neurons = [(seq_len/8),(seq_len/4),(seq_len/2)]
		param_grid = dict(learn_rate=learn_rate, neurons=neurons, seq_len = num_bases)

		#specify my own scorer for GridSearchCV that uses r2 instead of the estimator's scorer
		#try RandomizedSearchCV instead of GridSearchCV
		grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=make_scorer(my_custom_r2_func, greater_is_better=True), n_jobs=1)
		grid_result = grid.fit(xtrain, ytrain)


		# summarize results
		means = grid_result.cv_results_['mean_test_score']
		stds = grid_result.cv_results_['std_test_score']
		params = grid_result.cv_results_['params']
		for mean, stdev, param in zip(means, stds, params):
		    print("%f (%f) with: %r" % (mean, stdev, param))
		print("\n\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
		print "\nBest Estimator = ", grid_result.best_estimator_

		###################################################################################
		############ Need to extract best params and make a new model with it #############
		###################################################################################
		best_params = grid_result.best_params_
		tuned_model = create_model(best_params['learn_rate'], best_params['neurons'], best_params['seq_len'])
		tuned_model.fit(xtrain, ytrain, batch_size=128, nb_epoch=6, verbose=1)
		predicted = tuned_model.predict(xtest)

		slope, intercept, r_value, p_value, std_err = stats.linregress(ytest.reshape(-1),predicted.reshape(-1))
		print "R2 of tuned_model: ", r_value**2
		return tuned_model, predicted

	def __oneHotEncoder(self,seq):
		base_dict = {u'A':[1,0,0,0],u'C':[0,1,0,0],u'G':[0,0,1,0],u'T':[0,0,0,1]}
		return np.array([base_dict[x] for x in seq])

	def __oneHotDecoder(self,encseq):
		dec_seq = ""
		for x in encseq:
			if (x == np.array([1,0,0,0])).all():
				dec_seq += u'A'
			elif (x == np.array([0,1,0,0])).all():
				dec_seq += u'C'
			elif (x == np.array([0,0,1,0])).all():
				dec_seq += u'G'
			elif (x == np.array([0,0,0,1])).all():
				dec_seq += u'T'
		return dec_seq

	def __retrain(self):
		""" trains existing model """
		self.model.fit(self.X_train, self.Y_train, batch_size=128, nb_epoch=6, verbose=1)
		predicted = self.model.predict(self.X_test)
		slope, intercept, r_value, p_value, std_err = stats.linregress(self.Y_test.reshape(-1),predicted.reshape(-1))
		print "R2 of trained model: ", r_value**2
		self.save()
		return predicted

	def train(self):
		"""trains the current model, if a model filename exists.
		if it doesn't exist, then it builds an optimal model first
		by calling __opt_model which then trains that model"""

		if self.filename:
			print "Loading model: ", self.filename
			self.model = load_model(self.filename)
			self.predicted = self.__retrain()
		else:
			print "Making model. "
			self.model, self.predicted = self.__opt_model()
		
		

	def save(self):
		"""creates HDF5 file of the current model and saves in cur dir"""
		ts = time.time()
		st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
		self.filename = 'dnamodel'+st+'.h5'
		self.model.save(self.filename)  
		return 0

	def design(self):
		""" Currently returns a single optimized DNA sequence.

		TO DO: Returns a batch (list) of designs to test (default size = input len), ordered by
		expected output (max at the top) - outputs this to a text file."""

		
		df = self.df
		maxindx=df[['output1']].idxmax()
		


		#To DO: set some epsilon=<some small number> 
		#this will determine if optimization has saturated yet or not
		#continue generations until either epsilon or 50 generations is reached
		#(whichever comes first)
		new_seqs_list = []
		start_seq = self.__oneHotDecoder(self.X_test[self.predicted.argmax()])

		bases = [u'A',u'C',u'G',u'T']
		num_gen = 10

		
		for gen in range(0,num_gen):
			print "===== GENERATION ", gen, " ====="

			#taking max sequence from input data and mutating
			#to get 200 new sequences
			for j in range(0,200):
				base_idx = np.random.randint(0,self.seq_len-4)
				new_seq = list(start_seq)
				new_seq[base_idx] = np.random.choice(bases)
				new_seq[base_idx+1] = np.random.choice(bases)
				new_seq[base_idx+2] = np.random.choice(bases)
				strnew_seq = "".join(new_seq)
				if df.loc[df[u'sequence'] == strnew_seq].empty:
					new_seqs_list.append(strnew_seq)
			

			#format new sequences as CNN input for prediction
			Ztest = np.empty([len(new_seqs_list),self.seq_len,4])
			indx = 0
			for s in new_seqs_list:
				Ztest[indx] = self.__oneHotEncoder(s)
				indx += 1

			#CNN predicition
			Zpredicted = self.model.predict(Ztest)
			max_predicted_seq = new_seqs_list[Zpredicted.argmax()]
			
			#TO DO: might want to save all these mutated seqs instead of just 
			#keeping the max one, each generation
			new_seqs_list = []
			start_seq = max_predicted_seq
		print "\n\nBuild this sequence: ", max_predicted_seq, "to get this output: ", max(Zpredicted)

		return 0

	def test(self):
		"""this will move to test dir but for now just checking that a loaded .h5 file 
		predicts the same thing as the original model predicts (before saving)"""

		test_seqs = [u'A'*self.seq_len, u'C'*self.seq_len, u'G'*self.seq_len, u'T'*self.seq_len]

		##### format test sequences as CNN input for prediction ####
		Z_test = np.empty([len(test_seqs),self.seq_len,4])
		indx = 0
		for seq in test_seqs:
			Z_test[indx] = self.__oneHotEncoder(seq)
			indx += 1

		#print "current model", self.model.predict(Z_test).reshape(-1)
		#print "loaded model", load_model(self.filename).predict(Z_test).reshape(-1)
		#print self.model.predict(Z_test).reshape(-1)==load_model(self.filename).predict(Z_test).reshape(-1)





