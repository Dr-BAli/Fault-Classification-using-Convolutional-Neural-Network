# Import the libraries to be used in the program
import numpy as np
from keras.utils import plot_model
from scipy.io import loadmat
from keras.models import Model
from keras.layers import Input, Dense, LSTM
from keras.layers import Dropout, concatenate
from sklearn import datasets, metrics

#Load Training data and Labels For Supervised Training
trainx=loadmat('TrainX.mat')
x_tr=trainx['TrainX']
x_train=x_tr[:,0,:]
seqtrainx=loadmat('SeqTrain.mat')
seq_train=seqtrainx['SeqTrain']
trainy=loadmat('TrainY.mat')
y_train=trainy['TrainY']
#Load Testing data and Labels For Accuracy Calculation
testx=loadmat('TestX.mat')
x_ts=testx['TestX']
x_test=x_ts[:,0,:]
seqtestx=loadmat('SeqTest.mat')
seq_test=seqtestx['SeqTest']
testy=loadmat('TestY.mat')
y_test=testy['TestY']

main_input = Input(shape=(64, 3), name='main_input')
x1 = LSTM(64, return_sequences=True, input_shape=(64,3))(main_input)
x2 = LSTM(96, return_sequences=True)(x1)
lstm_out =LSTM(96)(x2)
auxiliary_input = Input(shape=(24,), name='aux_input')
x3 = concatenate([lstm_out, auxiliary_input])
x4 = Dense(128, activation='sigmoid', name='D1')(x3)
x5 = Dropout(0.3)(x4)
x6 = Dense(64, activation='sigmoid', name='D2')(x5)
x7 = Dropout(0.3)(x6)
main_output = Dense(1, activation='sigmoid', name='D3')(x7)
model = Model(inputs=[main_input, auxiliary_input], outputs=main_output)
plot_model(model, to_file='Final_LSTM_IEEE.png')
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
history = model.fit([seq_train, x_train],
          y_train, batch_size=64, epochs=250)
y_pred = model.evaluate([seq_test, x_test], y_test, verbose=1)
print('Test loss:', y_pred[0])
print('Test accuracy:', y_pred[1])
model.save('Final_LSTM_IEEE.h5')
