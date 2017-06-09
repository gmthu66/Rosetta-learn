import numpy as np 
from keras.models import Sequential, load_model
from keras.layers import Merge
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D
from keras.optimizers import RMSprop, SGD, Adam
from keras.wrappers.scikit_learn import KerasRegressor
from keras.regularizers import WeightRegularizer, l2 # import l2 regularizer
import random
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import preprocessing
from sklearn.metrics import make_scorer
import scipy.stats as stats
import datetime, time
from matplotlib import pyplot as plt
import seaborn as sea
import plotly.plotly as py
from sklearn.preprocessing import MinMaxScaler


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

def merge_model(loss = 'mean_squared_error', optimizer = 0, learn_rate = 0.001, l1_neurons = 90, l2_neurons = 20, l3_neurons = 10, in_len=93):
	cnn = Sequential()
	cnn.add(Convolution1D(nb_filter=30,filter_length=6,input_dim=20,input_length=43,border_mode="same", activation='relu'))
	cnn.add(Dropout(0.1))
	cnn.add(Convolution1D(nb_filter=40,filter_length=6,input_dim=20,input_length=43,border_mode="same", activation='relu'))
	cnn.add(Flatten())
	cnn.add(Dense(l1_neurons))
	cnn.add(Dropout(0.2))
	cnn.add(Activation('relu'))
	cnn.add(Dense(1))
	cnn.add(Activation('relu'))


	metrics = Sequential()
	metrics.add(Dense(90, activation='relu',input_dim=93))
	metrics.add(Dense(20, activation='relu'))
	metrics.add(Dense(1, activation='relu'))

	model = Sequential()
	model.add(Merge([cnn, metrics], mode='concat'))
	model.add(Dense(10))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('relu'))

	model.compile(loss=loss, optimizer='adam')

	return model



def conv_model(loss = 'mean_squared_error', optimizer = 0, learn_rate = 0.001, l1_neurons = 10, nb_filter = 40,filter_len1 = 6, filter_len2 = 6, in_len=66):
	cnn = Sequential()
	cnn.add(Convolution1D(nb_filter=30,filter_length=6,input_dim=21,input_length=in_len,border_mode="same", activation='relu'))
	cnn.add(Dropout(0.1))
	cnn.add(Convolution1D(nb_filter=40,filter_length=20,input_dim=21,input_length=in_len,border_mode="same", activation='relu'))

	cnn.add(Flatten())

	cnn.add(Dense(90))
	cnn.add(Dropout(0.2))
	cnn.add(Activation('relu'))

	cnn.add(Dense(10))
	cnn.add(Dropout(0.2))
	cnn.add(Activation('relu'))

	cnn.add(Dense(1))
	cnn.add(Activation('linear'))

	#compile the model
	adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	rms = RMSprop(lr=learn_rate, rho=0.9, epsilon=1e-08)

	cnn.compile(loss='mean_squared_error', optimizer=adam)

	return cnn

def ff_model(loss = 'mean_squared_error', optimizer = 0, learn_rate = 0.001, l1_neurons = 93, l2_neurons = 20, l3_neurons = 10, in_len=93):
	"""Builds a parameterized NN Model"""

	model = Sequential()
	model.add(Dense(l1_neurons, activation='relu',input_dim=in_len)) #, W_regularizer=WeightRegularizer(l1=0.001, l2=0.001)))))
	#model.add(Dense(1, activation='linear',input_dim=in_len)) #, W_regularizer=WeightRegularizer(l1=0.001, l2=0.001)))))
	#model.add(Dropout(0.30))
	model.add(Dense(l2_neurons, activation='relu'))
	# model.add(Dropout(0.25))
	# model.add(Dense(l3_neurons, activation='relu'))
	# model.add(Dropout(0.25))
	model.add(Dense(1, activation='relu'))

	#compile the model
	adam = Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	rms = RMSprop(lr=learn_rate, rho=0.9, epsilon=1e-08)
	#model.compile(loss='mean_squared_error', optimizer=adam)
	if optimizer == 0:
		model.compile(loss=loss, optimizer=rms)
	elif optimizer == 1:
		model.compile(loss=loss, optimizer='adam')
	elif optimizer == 2:
		model.compile(loss=loss, optimizer='sgd')

	return model

def my_custom_r2_func(ground_truth, predictions):
	slope, intercept, r_value, p_value, std_err = stats.linregress(ground_truth.reshape(-1),predictions.reshape(-1))
	print "..."
	return r_value**2
	


class proteinModel(object):
	""" An optimal CNN model for DNA sequences """

	def __init__(self, df, filename = ''):
		"""Initialize proteinModel object
		The input df should contain only two columns: sequence + expression"""

		self.filename = filename
		self.df = df
		self.seq_len, self.X_train, self.Y_train, self.X_test, self.Y_test, self.X2_train, self.X2_test = self.__parse_input()
		# self.seq_len, self.X_train, self.Y_train, self.X_test, self.Y_test = self.__parse_input()
		self.model = Sequential()
		self.model = Sequential()
		self.predicted = []
		
	def __parse_input(self):
		""" Splits training and test data from the input dataframe """
		#xtrain, ytrain, xtest, ytest, xtrain2, xvalid, ytrain2, yvalid = [], [], [], [], [], [], [], []
		xtrain, ytrain, xtest, ytest= [], [], [], []
		df = self.df
		############# Format NN inputs #############
		#seq_len = len(df['sequence'][0])
		#FIX THIS: DOnt keep it hardcoded
		seq_len = 66
		convX_data = np.empty([len(df),seq_len,21])
		# print len(df), len(df.iloc[1].values)
		ffX_data = np.empty([len(df),len(df.iloc[1].values)-1])
		#X_data = np.empty([len(df)])
		indx = 0

		Y_data = np.array(df[['output1']])

		#Metrics = df.iloc[:, :-1].values

		#for seq in df['sequence']:
		for seq in list(df.index.values):
			# print seq, df.loc[seq].values, Y_data[indx]
			convX_data[indx] = self.__oneHotEncoder(seq)
			ffX_data[indx] = df.loc[seq][:-1].values
			#X_data.append((self.__oneHotEncoder(seq), Metrics[indx]))
			indx += 1

		#zip up convX_data and ffX_data I think for the train test split?

		# print convX_data
		# print ffX_data
		# print Y_data

		# X_data = df.iloc[:, :-1].values
		# Y_data = df.iloc[:, -1:].values

		#print X_data, Y_data
		# print "convX_data : ", convX_data
		# print "ffX_data : ", ffX_data


		########## RANDOM TEST/TRAIN SPLIT #########
		#normed_out = preprocessing.StandardScaler().fit_transform(Y_data)
		#xtrain, xtest, ytrain, ytest = train_test_split(X_data, normed_out, test_size=0.15, random_state=42)
		xtrain, xtest, x2train, x2test, ytrain, ytest = train_test_split(convX_data, ffX_data, Y_data, test_size=0.15, random_state=42)
		#xtrain2, xvalid, ytrain2, yvalid = train_test_split(xtest, ytest, test_size=0.15, random_state=42)
		#return 43, xtrain, ytrain, xtest, ytest, xtrain2, xvalid, ytrain2, yvalid
		return 43, xtrain, ytrain, xtest, ytest, x2train, x2test

	def __opt_model(self):
		"""bulids a new model and optimizes hyperparams
		trains/fits and saves best model and params. 
		Currently a simple gridsearch, but should switch to a stochaistic 
		optimization alg with mean performance as the cost fcn """
		
		seq_len, xtrain, ytrain, xtest, ytest, x2train, x2test = self.seq_len, self.X_train, self.Y_train, self.X_test, self.Y_test, self.X2_train, self.X2_test
		#seq_len, xtrain, ytrain, xtest, ytest = self.seq_len, self.X_train, self.Y_train, self.X_test, self.Y_test

		#model = KerasRegressor(build_fn=create_model, nb_epoch=200, batch_size=16, verbose=0)
		print xtrain.shape, ytrain.shape
		#model = KerasRegressor(build_fn=conv_model, verbose=0)

		# define the grid search parameters
		num_metrics = [len(self.df.columns)-1]
		print 'number of metrics',num_metrics
		learn_rate = [0.001]
		#neurons = [(seq_len/8),(seq_len/4),(seq_len/2)]
		#l1_neurons = [90,70,50]
		#l2_neurons = [60,40,20,10]
		num_inputs = len(self.df.columns)-1
		l1_neurons = [10]
		#l1_neurons = [int((num_inputs+(num_inputs*0.20))), num_inputs, int((num_inputs+(num_inputs*0.20))/2)]
		#l2_neurons = [num_inputs, int((num_inputs+(num_inputs*0.20))/2)]
		l2_neurons = [30]
		l3_neurons = [20]
		filter_len1 = [20]
		filter_len2 = [20]
		nb_filter = [40]
		#l2_neurons = [90,60,30]
		#l3_neurons = [20,10]
		batch_size = [16]
		epochs = [20]
		optimizer = [1]
		loss = ['mean_squared_error']
		len_seq = [66]
		#param_grid = dict(batch_size=batch_size, nb_epoch=epochs, optimizer = optimizer, learn_rate=learn_rate, loss=loss, l1_neurons=l1_neurons, nb_filter=nb_filter, filter_len1=filter_len1, filter_len2=filter_len2, l2_neurons=l2_neurons, l3_neurons=l3_neurons, in_len = num_metrics)
		param_grid = dict(batch_size=batch_size, nb_epoch=epochs, optimizer = optimizer, learn_rate=learn_rate, loss=loss, l1_neurons=l1_neurons, nb_filter=nb_filter, filter_len1=filter_len1, filter_len2=filter_len2, in_len = len_seq)

		# #specify my own scorer for GridSearchCV that uses r2 instead of the estimator's scorer
		# #try RandomizedSearchCV instead of GridSearchCV
		# grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=make_scorer(my_custom_r2_func, greater_is_better=True), n_jobs=1)
		# grid_result = grid.fit(xtrain, ytrain)


		# # # summarize results
		# means = grid_result.cv_results_['mean_test_score']
		# stds = grid_result.cv_results_['std_test_score']
		# params = grid_result.cv_results_['params']
		# for mean, stdev, param in zip(means, stds, params):
		#     print("%f (%f) with: %r" % (mean, stdev, param))
		# print("\n\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
		# print "\nBest Estimator = ", grid_result.best_estimator_

		# ###################################################################################
		# ############ Need to extract best params and make a new model with it #############
		# ###################################################################################
		#best_params = grid_result.best_params_
		#tuned_model = conv_model(best_params['loss'], best_params['optimizer'], best_params['learn_rate'], best_params['l1_neurons'], best_params['l2_neurons'], best_params['l3_neurons'], best_params['in_len'])
		#######optimizer = 0, learn_rate = 0.001, l1_neurons = 10, nb_filter = 40,filter_len1 = 6, filter_len2 = 6, in_len=93
		#tuned_model = conv_model('mean_squared_error', 1, 0.001, 40, 40, 6, 6, 66)
		# tuned_model = conv_model(best_params['loss'], best_params['optimizer'], best_params['learn_rate'], best_params['nb_filter'], best_params['filter_len1'], best_params['filter_len2'], best_params['in_len'])
		tuned_model = conv_model('mean_squared_error', 1, 0.001, 40, 20, 20, 66)
		#tuned_model = merge_model('mean_squared_error', 1, 0.001, 40, 30, 20, 43)
		tuned_model.fit(xtrain, ytrain, nb_epoch=10, batch_size=16, verbose=1)
		#tuned_model.fit([xtrain,x2train], ytrain, nb_epoch=50, batch_size=16, verbose=1)
		#predicted = tuned_model.predict([xtest,x2test])
		predicted = tuned_model.predict(xtest)

		slope, intercept, r_value, p_value, std_err = stats.linregress(ytest.reshape(-1),predicted.reshape(-1))
		print "R2 of tuned_model: ", r_value**2

		d = {'y_pred': predicted.reshape(-1), 'y_actual': ytest.reshape(-1)}
		res_df = pd.DataFrame(data=d)

		sea.set(style="ticks", color_codes=True)
		g = sea.JointGrid(predicted.reshape(-1),ytest.reshape(-1)) #, xlim=(-3,3), ylim=(-3,3))
		g = g.plot_joint(plt.scatter, color='#3A12D5', edgecolor="white", alpha='0.1')
		
		g.ax_joint.set_xticks([0, 1, 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
		g.ax_joint.set_yticks([0, 1, 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
		sea.plt.show()


		############################################################
		######## Extra seq only test data from F.S. Dataset ########
		############################################################
		# rawfs = pd.read_excel('FS_test_set.xlsx', header=0)

		# fsdf = rawfs[np.isfinite(rawfs['min_stab'])]
		# #shuffle the dataframe here - have to do this here when sequences are a column
		# fsdf = fsdf.sample(frac=1)
		# #ADD BACK
		# fsdf = fsdf.set_index('full_sequence') 
		# #rescaling input to have mean 0 variance 1 - for efficient backprop
		# scaler = MinMaxScaler()
		# fin_fsdf = scaler.fit_transform(fsdf)
		# fin_fsdf = pd.DataFrame(data=fin_fsdf, index=fsdf.index, columns=fsdf.columns)

		# seq_len = 66
		# convX_sfdata = np.empty([len(fin_fsdf),seq_len,21])
		# sfY_data = np.array(fin_fsdf[['min_stab']])
		# indx = 0
		# for seq in list(fin_fsdf.index.values):
		# 	# print seq, df.loc[seq].values, Y_data[indx]
		# 	convX_sfdata[indx] = self.__oneHotEncoder(seq)
		# 	#ffX_data[indx] = df.loc[seq][:-1].values
		# 	#X_data.append((self.__oneHotEncoder(seq), Metrics[indx]))
		# 	indx += 1

		# sfpredicted = tuned_model.predict(convX_sfdata)

		# slope, intercept, sf_r_value, p_value, std_err = stats.linregress(sfY_data.reshape(-1),sfpredicted.reshape(-1))
		# print "R2 of tuned_model with FS's data: ", sf_r_value**2
		### the fit is 0.106 on the model trained with Gabe's data (yikesssss! - no agreement there)
		### does this mean that I had overfit? Or that FS's data just doesn't agree at all with GJR's data
		### can try training a model on FS's data and see how it fares there.


		########################## Validation ################################
		raw_eehee = pd.read_excel('GJR_cleaned_seq_only_eehee.xlsx', header=0)
		raw_ehee = pd.read_excel('GJR_cleaned_seq_only_ehee.xlsx', header=0)
		raw_heeh = pd.read_excel('GJR_cleaned_seq_only_heeh.xlsx', header=0)
		raw_hhh = pd.read_excel('GJR_cleaned_seq_only_hhh.xlsx', header=0)

		col_names = list(raw_eehee.columns)
		col_names[0] = u'sequence'
		col_names[-1] = u'output1'
		raw_eehee.columns = col_names
		#TO DO: should do this for each column
		df = raw_eehee[np.isfinite(raw_eehee['output1'])]
		raw_eehee = df.set_index('sequence')

		col_names = list(raw_ehee.columns)
		col_names[0] = u'sequence'
		col_names[-1] = u'output1'
		raw_ehee.columns = col_names
		#TO DO: should do this for each column
		df = raw_ehee[np.isfinite(raw_ehee['output1'])]
		raw_ehee = df.set_index('sequence')

		col_names = list(raw_heeh.columns)
		col_names[0] = u'sequence'
		col_names[-1] = u'output1'
		raw_heeh.columns = col_names
		#TO DO: should do this for each column
		df = raw_heeh[np.isfinite(raw_heeh['output1'])]
		raw_heeh = df.set_index('sequence') 

		col_names = list(raw_hhh.columns)
		col_names[0] = u'sequence'
		col_names[-1] = u'output1'
		raw_hhh.columns = col_names
		#TO DO: should do this for each column
		df = raw_hhh[np.isfinite(raw_hhh['output1'])]
		raw_hhh = df.set_index('sequence') 

		eeheeX_data = np.empty([len(raw_eehee),66,21])
		eheeX_data = np.empty([len(raw_ehee),66,21])
		heehX_data = np.empty([len(raw_heeh),66,21])
		hhhX_data = np.empty([len(raw_hhh),66,21])

		indx = 0
		for seq in list(raw_eehee.index.values):
			# print seq, df.loc[seq].values, Y_data[indx]
			eeheeX_data[indx] = self.__oneHotEncoder(seq)
			indx += 1

		indx = 0
		for seq in list(raw_ehee.index.values):
			# print seq, df.loc[seq].values, Y_data[indx]
			eheeX_data[indx] = self.__oneHotEncoder(seq)
			indx += 1

		indx = 0
		for seq in list(raw_heeh.index.values):
			# print seq, df.loc[seq].values, Y_data[indx]
			heehX_data[indx] = self.__oneHotEncoder(seq)
			indx += 1

		indx = 0
		for seq in list(raw_hhh.index.values):
			# print seq, df.loc[seq].values, Y_data[indx]
			hhhX_data[indx] = self.__oneHotEncoder(seq)
			indx += 1

		eehee1, eehee2 = train_test_split(eeheeX_data, test_size=0.5, random_state=42)
		heeh1, heeh2 = train_test_split(heehX_data, test_size=0.5, random_state=42)

		eehee_predicted = tuned_model.predict(eehee1)
		ehee_predicted = tuned_model.predict(eheeX_data)
		heeh_predicted = tuned_model.predict(heeh1)
		hhh_predicted = tuned_model.predict(hhhX_data)

		bins = np.arange(0, 1, .1) # fixed bin size
		plt.xlim([min(min(eehee_predicted),min(ehee_predicted),min(heeh_predicted),min(hhh_predicted))-.1, max(max(eehee_predicted),max(ehee_predicted),max(heeh_predicted),max(hhh_predicted))+.1])

		plt.hist(eehee_predicted, bins=bins, alpha=0.5, color = "red")
		plt.hist(hhh_predicted, bins=bins, alpha=0.5, color = "green")
		plt.hist(heeh_predicted, bins=bins, alpha=0.5, color = "purple")
		plt.hist(ehee_predicted, bins=bins, alpha=0.5, color = "blue")
		plt.hist([eehee_predicted,hhh_predicted,heeh_predicted,ehee_predicted], bins=bins, stacked=True, alpha=0.5, color = ["red","green","purple","blue"])
		plt.title('Predicted Topology Stabilities')
		plt.xlabel('variable X (bin size = .1)')
		plt.ylabel('count')
		plt.show()


		return tuned_model, predicted

	def __oneHotEncoder(self,seq):
		base_dict = {u'A':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
					 u'V':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
					 u'I':[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
					 u'L':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
					 u'P':[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
					 u'F':[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
					 u'W':[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
					 u'M':[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
					 u'G':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
					 u'S':[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
					 u'T':[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
					 u'C':[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
					 u'Y':[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
					 u'N':[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
					 u'Q':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
					 u'D':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
					 u'E':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
					 u'K':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
					 u'R':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
					 u'H':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
					 u'Z':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}

		# base_dict = {u'A':[1,0,0],
		# 			 u'V':[1,0,0],
		# 			 u'I':[1,0,0],
		# 			 u'L':[1,0,0],
		# 			 u'P':[1,0,0],
		# 			 u'F':[1,0,0],
		# 			 u'W':[1,0,0],
		# 			 u'M':[1,0,0],
		# 			 u'G':[0,1,0],
		# 			 u'S':[0,1,0],
		# 			 u'T':[0,1,0],
		# 			 u'C':[0,1,0],
		# 			 u'Y':[0,1,0],
		# 			 u'N':[0,1,0],
		# 			 u'Q':[0,1,0],
		# 			 u'D':[0,0,1],
		# 			 u'E':[0,0,1],
		# 			 u'K':[0,0,1],
		# 			 u'R':[0,0,1],
		# 			 u'H':[0,0,1]}
		return np.array([base_dict[x] for x in seq])

	def __oneHotDecoder(self,encseq):
		dec_seq = ""
		for x in encseq:
			if (x == np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
				dec_seq += u'A'
			elif (x == np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
				dec_seq += u'V'
			elif (x == np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
				dec_seq += u'I'
			elif (x == np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
				dec_seq += u'L'
			elif (x == np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
				dec_seq += u'P'
			elif (x == np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
				dec_seq += u'F'
			elif (x == np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
				dec_seq += u'W'
			elif (x == np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
				dec_seq += u'M'
			elif (x == np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])).all():
				dec_seq += u'G'
			elif (x == np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])).all():
				dec_seq += u'S'
			elif (x == np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])).all():
				dec_seq += u'T'
			elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])).all():
				dec_seq += u'C'
			elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])).all():
				dec_seq += u'Y'
			elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])).all():
				dec_seq += u'N'
			elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])).all():
				dec_seq += u'Q'
			elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])).all():
				dec_seq += u'D'
			elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])).all():
				dec_seq += u'E'
			elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])).all():
				dec_seq += u'K'
			elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])).all():
				dec_seq += u'R'
			elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])).all():
				dec_seq += u'H'
			elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])).all():
				dec_seq += u'Z'
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
		self.filename = 'proteinmodel'+st+'.h5'
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





