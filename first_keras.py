# -*- coding: utf-8 -*-


import tensorflow # Imports tensorflow
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


dataset = loadtxt("pima-indians-diabetes.csv", delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]
print(X[0])
print(Y)

train_X = X[:500]


test_X = X[501:767]


train_Y = Y[:500]
test_Y = Y[501:767]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(train_X,train_Y,epochs=150, batch_size=10)

model.evaluate(test_X, test_Y, batch_size=10)

