# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import tensorflow as tf
import tensorflow.keras.datasets.cifar10
import tensorflow.keras.utils.to_categorical

import tensorflow.keras.models.Sequential
import tensorflow.keras.layers
# .Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
import tensorflow.keras.callbacks.EarlyStopping
import tensorflow.keras.preprocessing.image.ImageDataGenerator

import sklearn.metrics.ConfusionMatrixDisplay
import sklearn.metrics.classification_report
import sklearn.metrics.confusion_matrix
import tensorflow.keras.applications.densenet.DenseNet121

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

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

# Scale the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Transform target variable into one-hotencoding
y_cat_train = to_categorical(y_train, 10)
y_cat_test = to_categorical(y_test, 10)

y_cat_train

# Load pre-trained DenseNet model with frozen convolutional layers
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(32,32,3))
for layer in base_model.layers:
    layer.trainable = False

# Add fully connected layers on top of frozen convolutional layers
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu', input_shape=(512,))(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)

# Add final classification layer
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

# Create final model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Print the accuracy
accuracy = model.evaluate(x_test, y_test)[1]
print(f'Test accuracy: {accuracy}')

# Load pre-trained DenseNet model with frozen convolutional layers
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(32,32,3))
for layer in base_model.layers:
    layer.trainable = False

# Add fully connected layers on top of frozen convolutional layers
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu', input_shape=(512,))(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)

# Add final classification layer
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

# Create final model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Plot the evolution of the accuracy
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy evolution during training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Print the accuracy
accuracy = model.evaluate(x_test, y_test)[1]
print(f'Test accuracy: {accuracy}')

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Load pre-trained DenseNet model with frozen convolutional layers
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(32,32,3))
for layer in base_model.layers:
    layer.trainable = False

# Add fully connected layers on top of frozen convolutional layers
x = base_model.output
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu', input_shape=(512,))(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)

# Add final classification layer
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

# Create final model
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Define the learning rate range
lr_range = [0.0001, 0.001, 0.01, 0.1]

# Train the model for 10 epochs with different learning rates
accuracies = []
for lr in lr_range:
    model.optimizer.lr = lr
    history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))
    accuracies.append(history.history['val_accuracy'][-1])

# Plot the test accuracy vs. the learning rates
plt.plot(lr_range, accuracies, 'o-')
plt.xscale('log')
plt.title('Test accuracy vs. learning rate')
plt.xlabel('Learning rate')
plt.ylabel('Test accuracy')
plt.show()

# Print the accuracy
accuracy = model.evaluate(x_test, y_test)[1]
print(f'Test accuracy: {accuracy}')