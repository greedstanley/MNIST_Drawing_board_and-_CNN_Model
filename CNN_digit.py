import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt



train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X_train, X_test, y_train, y_test = train_test_split(train_data.iloc[:,1:], 
                                                   train_data.iloc[:,0],
                                                   test_size=0.2)

plt.figure(figsize=(16,6))
for i in range(20):
    plt.subplot(4,10,i+1)
    plt.axis("off")
    plt.title("label " + str(train_data.iloc[i,0]))
    plt.imshow(train_data.iloc[i,1:].values.reshape(28,28), cmap="gray")
plt.tight_layout()

standscalar = StandardScaler()
standscalar.fit(X_train)
X_train_DNN = standscalar.transform(X_train)
X_test_DNN = standscalar.transform(X_test)

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import Adam

#%% DNN model


DNNmodel_0 = Sequential([
    Dense(300, activation = "relu", input_shape=(784,)),
    Dense(100, activation = "relu"),
    Dense(10, activation = "softmax")
    ])
DNNmodel_0.compile(loss = "categorical_crossentropy",
                 optimizer = "Adam",
                 metrics = ["accuracy"])
DNNmodel_0.summary()
history_0 = DNNmodel_0.fit(X_train_DNN,y_train,
                           batch_size=45,
                           epochs = 5,
                           validation_data=(X_test,y_test))


DNNmodel_1 = Sequential([
    Dense(300, activation='relu', input_shape = (784,)),
    Dropout(0.2),
    Dense(100, activation = 'relu'),
    Dropout(0.2),
    Dense(10,activation = 'softmax')    ])

DNNmodel_1.compile(loss = 'categorical_crossentropy',
                   optimizer = 'Adam',
                   metrics = ['accuracy'])

DNNmodel_1.summary()

history_1 = DNNmodel_1.fit(X_train_DNN, y_train,
                           batch_size = 50,
                           epochs = 5,
                           validation_data=(X_test,y_test))

#%% CNN model
from keras.layers import Convolution2D, MaxPooling2D, Flatten

X_train_CNN = X_train.values.reshape(-1,28,28,1)
X_test_CNN = X_test.values.reshape(-1,28,28,1)


CNNmodel = Sequential([
    Convolution2D(32, (3,3), strides=(1,1), padding='same', activation='relu', input_shape = (28,28,1)),
    Convolution2D(32, (3,3), strides=(1,1), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2,2), padding='same'),
    Dropout(0.25),
    Convolution2D(64, (3,3), strides=(1,1), padding='same', activation='relu'),
    Convolution2D(64, (3,3), strides=(1,1), padding='same', activation='relu'),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(10, activation="softmax")     ])

CNNmodel.compile(loss = 'categorical_crossentropy',
                 optimizer='adam',
                 metrics = ['accuracy'])

CNNmodel.summary()

CNN_history = CNNmodel.fit(X_train_CNN,y_train,
                           batch_size=30,
                           epochs = 5,
                           validation_data=(X_test_CNN,y_test))

test_data = standscalar.transform(test_data)
submit_test = test_data.reshape(-1,28,28,1)
result = CNNmodel.predict(submit_test)

answer = pd.DataFrame()
answer["ImageId"] = np.arange(1,len(result)+1,1)
answer["Label"] = np.argmax(result, axis=1)
answer.to_csv("sumbit.csv")


from tensorflow import keras
CNNmodel.save("mnist_dect_CNN_model.h5")
new_CNNmodel = keras.models.load_model("mnist_dect_CNN_model.h5")
    
    
