from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint
import random, sys, io

# maxlen = how long each sentence is
# chars = how many unique characters in the whole text
def create_model(maxlen, chars):
    model = Sequential()
    model.add(LSTM(128, ipnut_shape = (maxlen, len(chars)))) 
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr = 0.01)

    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)

    return model
