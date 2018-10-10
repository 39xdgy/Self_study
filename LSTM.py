'''
https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.optimizers import RMSprop, Adam
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.utils import np_utils
import random, sys, io

# maxlen = how long each sentence is
# chars = how many unique characters in the whole text

def read_input(input_path):
    with open(input_path, 'r') as f:
        input_english = f.read()

    input_english = input_english.lower()
    #print(input_english)

    chars = sorted(list(set(input_english)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    #print(chars)
    #print(char_to_int)

    n_chars = len(input_english)
    n_vocab = len(chars)
    #print(n_chars)
    #print(n_vocab)
    

    return input_english, chars, char_to_int, n_chars, n_vocab




def create_model(seq_length, n_vocab):
    model = Sequential()
    model.add(LSTM(128, input_shape = (seq_length, 1), return_sequences = False))
    #model.add(Dropout(0.2))
#    model.add(LSTM(256))
    #model.add(Dropout(0.2))
    model.add(Dense(n_vocab, activation = 'softmax'))

    optimizer = RMSprop(lr = 0.01)

    model.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    return model




raw_text, chars, char_to_int, n_chars, n_vocab = read_input("./11-0.txt")

#print(raw_text, "\n", chars, "\n", char_to_int, "\n", n_chars, "\n", n_vocab)


seq_length = 100
dataX, dataY = [], []

for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

#print(dataX)

X = np.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
y = np_utils.to_categorical(dataY)
print(len(X))
model = create_model(seq_length, n_vocab)

filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor = 'loss', verbose = 1, save_best_only = True, mode = 'min')
callbacks_list = [checkpoint]


model.fit(X, y, epochs = 10, batch_size = 512, callbacks = callbacks_list)

int_to_char = dict((i, c) for i, c in enumerate(chars))

start = np.random.randint(0, len(dataX) - 1)
pattern = dataX[start]
print ("Seed:")
print ("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x/float(n_vocab)
    prediction = model.predict(x, verbose = 0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone. ")
