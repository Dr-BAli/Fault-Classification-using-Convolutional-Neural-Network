# Import the libraries to be used in the program
import numpy as np
import scipy.io as sci
from keras.utils import plot_model
from scipy.io import loadmat
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, GRU
from keras.layers import Dropout, concatenate
from sklearn import datasets, metrics
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix 

trainx=loadmat('NDZtst4.mat')
x_tr=trainx['NDZtst4']
x_test=x_tr[:,0,:]
seqtrainx=loadmat('SeqNDZ4.mat')
seq_test=seqtrainx['SeqNDZ4']
del model
model = load_model('Final_LSTM_IEEE.h5')

pred_ndz4=model.predict([seq_test, x_test]) 

sci.savemat('D:\IEEE_islanding Detection\Simulation_islanding detection\Matlab\pred_ndz4.mat', mdict={'pred_ndz4': pred_ndz4})
