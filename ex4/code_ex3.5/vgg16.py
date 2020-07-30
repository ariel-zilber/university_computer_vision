import keras
from keras.models import Sequential
from keras.layers import Dense,  Dropout, Flatten, ZeroPadding2D
from keras.layers import Conv2D, Input
from keras.layers import MaxPooling2D
from keras.optimizers import Adam

LEARNING_RATE=0.001

def vgg16():
    '''
    Implementation of VGG16 based on :https://arxiv.org/pdf/1409.1556
    Modified for MNIST dataset  0/1 handwritten digits binary learning
    :return:
    '''
    model = Sequential()

    model.add(Input(shape=(28, 28, 1)))
    model.add(ZeroPadding2D(padding=(2, 2), input_shape=(28, 28, 1)))

    model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3),padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3),padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, kernel_size=(3, 3),padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu',name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(units=128,activation="relu",name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid',name='fc3'))
    opt = Adam(lr=LEARNING_RATE)

    model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

    return model

