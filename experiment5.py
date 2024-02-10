# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
#Given an image, can we predict the correct class of this image?

# The images are very small (32x32) and by visualizing them you will notice how difficult it is to distinguish them even for a human.

# In this notebook we are going to build a CNN model that can classify images of various objects. We have 10 class of images:

# Airplane
# Automobile
# Bird
# Cat
# Deer
# Dog
# Frog
# Horse
# Ship
# Truck

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, confusion_matrix


#Load the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

"""Data Visualization"""

# Define the labels of the dataset
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

# Let's view more images in a grid format
# Define the dimensions of the plot grid
W_grid = 10
L_grid = 10

# fig, axes = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object
# we can use the axes object to plot specific figures at various locations

fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))

axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

n_train = len(X_train) # get the length of the train dataset

# Select a random number from 0 to n_train
for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables

    # Select a random number
    index = np.random.randint(0, n_train)
    # read and display an image with the selected index
    axes[i].imshow(X_train[index,1:])
    label_index = int(y_train[index])
    axes[i].set_title(labels[label_index], fontsize = 8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)

"""Data Preprocessing"""

# Scale the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Transform target variable into one-hotencoding
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

y_cat_train

"""CNN Model"""

# Set the shape of the input data
INPUT_SHAPE = (32, 32, 3)

# Set the size of the convolutional kernel
KERNEL_SIZE = (3, 3)

# Initialize a sequential model
model = Sequential()

# Add a convolutional layer with 32 filters, ReLU activation, and same padding
model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))

# Add batch normalization layer
model.add(BatchNormalization())

# Add another convolutional layer with 32 filters, ReLU activation, and same padding
model.add(Conv2D(filters=32, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))

# Add another batch normalization layer
model.add(BatchNormalization())

# Add a max pooling layer with a pool size of (2,2)
model.add(MaxPool2D(pool_size=(2, 2)))

# Add a dropout layer with a rate of 0.25
model.add(Dropout(0.25))

# Add a convolutional layer with 64 filters, ReLU activation, and same padding
model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))

# Add batch normalization layer
model.add(BatchNormalization())

# Add another convolutional layer with 64 filters, ReLU activation, and same padding
model.add(Conv2D(filters=64, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))

# Add another batch normalization layer
model.add(BatchNormalization())

# Add a max pooling layer with a pool size of (2,2)
model.add(MaxPool2D(pool_size=(2, 2)))

# Add a dropout layer with a rate of 0.25
model.add(Dropout(0.25))

# Add a convolutional layer with 128 filters, ReLU activation, and same padding
model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))

# Add batch normalization layer
model.add(BatchNormalization())

# Add another convolutional layer with 128 filters, ReLU activation, and same padding
model.add(Conv2D(filters=128, kernel_size=KERNEL_SIZE, input_shape=INPUT_SHAPE, activation='relu', padding='same'))

# Add another batch normalization layer
model.add(BatchNormalization())

# Add a max pooling layer with a pool size of (2,2)
model.add(MaxPool2D(pool_size=(2, 2)))

# Add a dropout layer with a rate of 0.25
model.add(Dropout(0.25))

# Flatten the output of the previous layer
model.add(Flatten())

# Add a dense layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='relu'))

# Add a dropout layer with a rate of 0.25
model.add(Dropout(0.25))

# Add a dense layer with 10 neurons and softmax activation
model.add(Dense(10, activation='softmax'))

# Define the metrics to be used during model training and evaluation
METRICS = [
    'accuracy',
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

# Compile the model with categorical crossentropy loss, Adam optimizer, and the defined metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS)

model.summary()

"""Early stopping"""

early_stop = EarlyStopping(monitor='val_loss', patience=2)

"""Data Augmentations"""

# Define the batch size for data augmentation
batch_size = 32

# Define an ImageDataGenerator object for data augmentation
data_generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# Generate batches of augmented data using the ImageDataGenerator object
train_generator = data_generator.flow(X_train, y_cat_train, batch_size)

# Calculate the number of steps (batches) per epoch
steps_per_epoch = X_train.shape[0] // batch_size

# Train the model using the augmented data generator
# Fit the model to the data generator for 50 epochs and track the validation accuracy
# Set the steps_per_epoch argument to the calculated number of steps
# Set the validation_data argument to the test data
r = model.fit(train_generator,
              epochs=5,
              steps_per_epoch=steps_per_epoch,
              validation_data=(X_test, y_cat_test))

plt.figure(figsize=(12, 16))

plt.subplot(4, 2, 1)
plt.plot(r.history['loss'], label='Loss')
plt.plot(r.history['val_loss'], label='val_Loss')
plt.title('Loss Function Evolution')
plt.legend()

plt.subplot(4, 2, 2)
plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy Function Evolution')
plt.legend()

plt.subplot(4, 2, 3)
plt.plot(r.history['precision'], label='precision')
plt.plot(r.history['val_precision'], label='val_precision')
plt.title('Precision Function Evolution')
plt.legend()

plt.subplot(4, 2, 4)
plt.plot(r.history['recall'], label='recall')
plt.plot(r.history['val_recall'], label='val_recall')
plt.title('Recall Function Evolution')
plt.legend()

"""Experiment 5 - Note 2"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model with a learning rate scheduler
initial_learning_rate = 0.01
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model for 10 epochs
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# Plot the learning rate vs. test accuracy
learning_rates = initial_learning_rate * np.power(0.96, np.arange(10))
test_accs = history.history['val_accuracy']
plt.plot(learning_rates, test_accs)
plt.xscale('log')
plt.xlabel('Learning rate')
plt.ylabel('Test accuracy')
plt.title('Learning Rate Schedule')
plt.show()

# Evaluate the model on the test set and print the accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')