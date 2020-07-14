import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, Sequential
from keras.layers import Flatten, LSTM, Activation, Dense, Dropout, Input, Embedding, GRU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.sequence import pad_sequences

train = pd.read_csv("../Data/preprocessed_training_data.csv", encoding="utf-8")
test = pd.read_csv("../Data/preprocessed_testing_data.csv", encoding="utf-8")
Train_X, Test_X, Train_Y, Test_Y = train.text, test.text, train.choose_one, test.choose_one
 
max_words = 15000
max_len = 50
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(Train_X)
sequences = tok.texts_to_sequences(Train_X)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

test_sequences = tok.texts_to_sequences(Test_X)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# EMBEDDING_DIM = 64


def RNN_LSTM():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,64,input_length=max_len)(inputs)
    layer = Dropout(0.5)(layer)
    layer = LSTM(100)(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    model.fit(sequences_matrix,Train_Y,batch_size=128,epochs=10,
              validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
    
    accr = model.evaluate(test_sequences_matrix,Test_Y)
    
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

def CNN_LSTM():
    model = Sequential()
    model.add(Embedding(max_words,32,input_length=max_len))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
    model.fit(sequences_matrix,Train_Y,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
    
    accr = model.evaluate(test_sequences_matrix,Test_Y)
    
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    
def GRU_LSTM():
    model = Sequential()
    model.add(Embedding(max_words, 100, input_length=max_len))
    model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
    model.fit(sequences_matrix, Train_Y, batch_size=128, epochs=10, validation_data=(test_sequences_matrix, Test_Y), verbose=2)
    
    accr = model.evaluate(test_sequences_matrix,Test_Y)
    
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    
    
# estimator = KerasClassifier(build_fn=RNN, epochs=5, batch_size=100, verbose=0)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# results = cross_val_score(estimator, sequences_matrix, Train_Y, cv=kfold)
# results.mean()*100, results.std()*100

# Run models
#RNN_LSTM()
#CNN_LSTM()
#GRU_LSTM()