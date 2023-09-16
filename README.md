# Convolutional Deep Neural Network for Digit Classification

## AIM:

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset:
Digit classification and to verify the response for scanned handwritten images.

The MNIST dataset is a collection of handwritten digits. The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. Here we build a convolutional neural network model that is able to classify to it's appropriate numerical value.

## Neural Network Model:

![image](https://user-images.githubusercontent.com/93427246/230703865-a6518aa2-7d2d-41fe-ae44-696ecbe3fc2e.png)

## DESIGN STEPS:

### STEP 1:
Import tensorflow and preprocessing .
### STEP 2:
Build a CNN model.
### STEP 3:
Compile and fit the model and then predict.


## PROGRAM:
## Developed By: Vineesh.M
## Reg No: 212221230122
```py
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
inputs = keras.Input(shape=(28,28,1))
model.add(inputs)
model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')) 
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(20,activation='relu'))
model.add(layers.Dense(13,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))

metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()

print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))


img = image.load_img('eight.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)
```

## OUTPUT:

### Training Loss, Validation Loss Vs Iteration Plot:
![image](https://user-images.githubusercontent.com/93427246/230704047-af8c3028-5dbf-4c16-b336-fa4b2c44bfa2.png)
![image](https://user-images.githubusercontent.com/93427246/230704054-f99d26eb-cae4-4111-99ed-eb5eda2b8321.png)


### Classification Report:

![image](https://user-images.githubusercontent.com/93427246/230704067-2e2d3927-8ac8-49c0-9087-8ecc3a6c9d50.png)

### Confusion Matrix:
![image](https://user-images.githubusercontent.com/93427246/230704078-66950bd4-a942-43e6-b93e-e31fa2ef2b16.png)

### New Sample Data Prediction:
![image](https://user-images.githubusercontent.com/93427246/230704109-a8eccefc-2d83-48f5-a3f8-a6e3daffce57.png)

## RESULT:
A convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is developed sucessfully.
