from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
import os.path
from os import path
from vgg16 import vgg16
FEATURES_LAYERS=13
WEIGHTS_PATH="MNIST_VGG16.h5"

class FeatureExtractor():
    '''
    Extract features
    '''

    def __init__(self,x_train,y_train,batch_size,epochs,x_test,y_test):
        self.x_train=x_train
        self.y_train=y_train
        self.batch_size=batch_size
        self.epochs=epochs
        self.x_test=x_test
        self.y_test=y_test
        self.model=vgg16()

    def fit(self):
        if path.exists(WEIGHTS_PATH):
            self.model.load_weights(WEIGHTS_PATH)
        else:
            history = self.model.fit(self.x_train, self.y_train,
              batch_size=self.batch_size,
              epochs=self.epochs,
            verbose=1,
            validation_data=(self.x_test, self.y_test))

            self.model.save_weights(WEIGHTS_PATH)

        # extract features from last convolution layer of VGG16
        self.model.pop()
        self.model.pop()
        self.model.pop()
        self.model.pop()
        self.model.pop()
        self.model.pop()

    def extract_features(self,data):
        return self.model.predict(data)

