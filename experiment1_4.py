# -*- coding: utf-8 -*-
"""Experiment1-4.ipynb
"""

import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

"""##Load and Preprocess data"""

train = CIFAR10(root = './data', train = True, download = True)
test = CIFAR10(root = './data', train = False, download = True)

x_train = []
x_test = []
y_train = []
y_test = []

for i in range(len(train)):
    x_train.append(np.asarray(train[i][0]))
    y_train.append(train[i][1])
    i += 1

for i in range(len(test)):
    x_test.append(np.asarray(testnset[i][0]))
    y_test.append(test[i][1])
    i += 1

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)


x_nonorm_train = x_train # Save unnormalized data

#Normalize Data
x_train = x_train / 255.0

# Number of features:
feat = 1
for d in x_train[0].shape:
    feat *= d

# Flattening the data:
x_train = x_train.reshape(-1, feat, 1)
x_nonorm_train = x_nonorm_train.reshape(-1, feat, 1)

x_train = np.squeeze(x_train)
x_nonorm_train = np.squeeze(x_nonorm_train)
y_train = np.squeeze(y_train)
y_test = np.squeeze(y_test)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

"""##MLP"""

class Layer:

    def __init__(self, layer_type='D', I = 256, O = 256, gamma = 0):
        self.layer_type = layer_type
        self.I = I  # input dimensions
        self.O = O # output dimensions
        self.gamma = gamma

        if layer_type == 'D':
          # Initialize weights and biases
          self.W = np.random.normal(loc = 0.0, scale = np.sqrt(2 / (I + O)), size = (I ,O))
          # Biases:
          self.b = np.zeros(O)

    def feedforward(self, x):
      if self.layer_type == 'R':
        return np.maximum(x, 0)   # Relu activation
      elif self.layer_type == 'D':
        return np.dot(x, self.W) + self.b   # Dense layer
      elif self.layer_type == 'LR':  # Leaky relu activation
        return np.maximum(x, 0) + self.gamma * np.minimum(x, 0)
      elif self.layer_type == 'T':   # Tanh activation
        return np.tanh(x)

    def backpropagation(self, x, grad_y, learning_rate, l2_reg, l1_reg):
      if self.layer_type == 'R':
        return grad_y * (x > 0)  # Relu derivative * gradient of previous layer

      elif self.layer_type == 'D':
        grad_x = np.dot(grad_y, self.W.T)

        grad_W = np.dot(x.T, grad_y)
        grad_b = grad_y.mean(axis = 0) * x.shape[0]

        # Update weights
        self.W = self.W - learning_rate * grad_W - l2_reg * self.W - l1_reg * np.sign(self.W)
        self.b = self.b - learning_rate * grad_b

        return grad_x

      elif self.layer_type == 'LR':  # Leaky relu derivative * gradient of previous layer
        return grad_y * ((x > 0) + self.gamma * (x < 0))
      elif self.layer_type == 'T':   # Tanh derivative * gradient of previous layer
        return grad_y * (1 - np.square(np.tanh(x)))

class MLP:

    def __init__( self, layers, epochs = 50, batch_size = 40, epsilon = 1e-6, lr = 0.01, l2_reg=0, l1_reg=0):

        self.layers = layers
        self.epochs = epochs
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.lr = lr
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.losses = []

    def softmax(self, z):
        return np.exp(z) / np.exp(z).sum(axis = -1, keepdims = True)

    # Softmax crossentropy loss
    def loss(self, z, y ):
        return - z[np.arange(len(z)), y] + np.log(np.sum(np.exp(z), axis = -1))

    # Get dy
    def grad_loss(self, z, y):

        yh = self.softmax(z)

        ones = np.zeros_like(z)
        ones[np.arange(len(z)), y] = 1

        return (- ones + yh) / z.shape[0]

    def feedforward(self, layers,  x):

        activated_outputs = [] # contains [z1, z2, yh]

        for layer in layers:
            activated_outputs.append(layer.feedforward(x))
            x = activated_outputs[-1]

        return activated_outputs

    def get_minibatches(self, X, y, batch_size):
        batches_x = []
        batches_y = []
        num_batches = 0
        indices = np.random.permutation(len(X))
        for i in range(0, len(X) - batch_size + 1, batch_size):
            j = indices[i:i + batch_size]
            batches_x.append(X[j])
            batches_y.append(y[j])
            num_batches += 1

        return batches_x, batches_y, num_batches

    def fit(self, x, y): #Creates minibatches and fits model
      batches_x, batches_y, num_batches = self.get_minibatches(x, y, self.batch_size)
      t = 0
      while t < self.epochs:
        for i in range(num_batches):
          batchx = batches_x[i].reshape(-1, 3072)
          batchy = batches_y[i].reshape(batchx.shape[0], )
          loss = self.train(batchx, batchy)
          if i == 0:
            self.losses.append(loss)
        t += 1

    # Performs forward and backward pass on given set of data
    def train(self, X, y):

        yh = self.feedforward(self.layers, X)
        layer_inputs = [X] + yh
        uc = yh[-1]
        y = y.astype(int)
        dy = self.grad_loss(uc, y)

        # Backpropagation:
        for i in range(len(self.layers))[::-1]:
            layer = self.layers[i]
            dy = layer.backpropagation(layer_inputs[i], dy, self.lr, self.l2_reg, self.l1_reg)
        loss = self.loss(uc, y)
        return np.mean(loss)

    def predict(self, X):
        z = self.feedforward(self.layers, X)[-1]
        return z.argmax(axis = -1)

    def evaluate_acc(self, yh, y):
      return np.mean(yh == y)

"""## Experiments

Accuracy Comparison - All 3 models
"""

# Build the three models

# No hidden layers
layers1 = [Layer(I = x_train.shape[1], O = 10)]
model1 = MLP(layers=layers1)

# One hidden layer
layers2 = [Layer(I=x_train.shape[1]), Layer(layer_type='R'), Layer(O=10)]
model2 = MLP(layers=layers2)


# Two hidden layers
layers3 = [Layer(I=x_train.shape[1]), Layer(layer_type='R'), Layer(), Layer(layer_type='R'), Layer(O=10)]
model3 = MLP(layers = layers3)

# Determine Accuracies

model1.fit(x_train, y_train)
yh1_test = model1.predict(x_test)
acc1_test = model1.evaluate_acc(yh1_test, y_test)
print("MLP1: No Hidden Layer Test Accuracy:"  + str(acc1_test))

model2.fit(x_train, y_train)
yh2_test = model2.predict(x_test)
acc2_test = model2.evaluate_acc(yh2_test, y_test)
print("MLP2: Single Hidden Layer Test Accuracy:"  + str(acc2_test))

model3.fit(x_train, y_train)
yh3_test = model3.predict(x_test)
acc3_test = model3.evaluate_acc(yh3_test, y_test)
print("MLP3: Two Hidden Layers Test Acc:" + str(acc3_test))

"""## 2 Hidden Layer"""

# Divide training set into training and validation

x_train1, x_valid = x_train[:-10000], x_train[-10000:]
y_train1, y_valid = y_train[:-10000], y_train[-10000:]

"""###Best Leaky Relu Gamma"""

# Determine gamma that gives highest validation accuracy


gammas = [0.0001, 0.001, 0.01, 0.1]
accuracies = []

for i in range(len(gammas)):
  layers_leaky = [Layer(I=x_train.shape[1]), Layer(layer_type='LR', gamma=gammas[i]), Layer(), Layer(layer_type='LR', gamma=gammas[i]),Layer(O=10)]
  model = MLP(layers=layers_leaky)
  model.fit(x_train1, y_train1)
  yh = model.predict(x_valid)
  acc = model.evaluate_acc(yh, y_valid)
  accuracies.append(acc)


fig = plt.figure()
plt.plot(gammas, accuracies, marker= '.', alpha=.998)
plt.xlabel('Gamma')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy as a function of gamma - Two hidden layers, Leaky Relu')
plt.show()

"""###Batch size"""

# Determine batch size that gives highest validation accuracy


batch_sizes = [40, 100, 300, 700, 1000]
accs_r = []
accs_lr = []
accs_t = []


for i in range(len(batch_sizes)):
  layers_relu = [Layer(I=x_train.shape[1]), Layer(layer_type='R'), Layer(), Layer(layer_type='R'),Layer(O=10)]
  layers_tanh = [Layer(I=x_train.shape[1]), Layer(layer_type='T'), Layer(), Layer(layer_type='T'),Layer(O=10)]
  layers_leaky = [Layer(I=x_train.shape[1]), Layer(layer_type='LR', gamma=0.001), Layer(), Layer(layer_type='LR', gamma=0.001),Layer(O=10)]
  modelr = MLP(layers_relu, batch_size=batch_sizes[i], epochs=10)
  modellr = MLP(layers_leaky, batch_size=batch_sizes[i], epochs=10)
  modelt = MLP(layers_tanh, batch_size = batch_sizes[i],epochs=10)
  modelr.fit(x_train1, y_train1)
  modellr.fit(x_train1, y_train1)
  modelt.fit(x_train1, y_train1)
  yhr = modelr.predict(x_valid)
  yhlr = modellr.predict(x_valid)
  yht = modelt.predict(x_valid)
  accs_r.append(modelr.evaluate_acc(yhr, y_valid))
  accs_lr.append(modellr.evaluate_acc(yhlr, y_valid))
  accs_t.append(modelt.evaluate_acc(yht, y_valid))


fig = plt.figure()
plt.plot(batch_sizes, accs_r, marker= '.', color="blue", alpha=.998, label="Relu")
plt.plot(batch_sizes, accs_lr, marker= '.', color="red", alpha=.998, label="Leaky Relu")
plt.plot(batch_sizes, accs_t, marker= '.', color="green", alpha=.998, label = "Tanh")
plt.xlabel('Batch Size')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy as a function of batch size')
plt.legend()
plt.show()

"""### Learning Rate"""

# Determine batch size that gives highest validation accuracy


learning_rates = [0.001, 0.01, 0.1]
accs_r = []
accs_lr = []
accs_t = []


for i in range(len(learning_rates)):
  layers_relu = [Layer(I=x_train.shape[1]), Layer(layer_type='R'), Layer(), Layer(layer_type='R'),Layer(O=10)]
  layers_tanh = [Layer(I=x_train.shape[1]), Layer(layer_type='T'), Layer(), Layer(layer_type='T'),Layer(O=10)]
  layers_leaky = [Layer(I=x_train.shape[1]), Layer(layer_type='LR', gamma=0.001), Layer(), Layer(layer_type='LR', gamma=0.001),Layer(O=10)]
  modelr = MLP(layers_relu, lr = learning_rates[i])
  modellr = MLP(layers_leaky, lr = learning_rates[i])
  modelt = MLP(layers_tanh, lr = learning_rates[i])
  modelr.fit(x_train1, y_train1)
  modellr.fit(x_train1, y_train1)
  modelt.fit(x_train1, y_train1)
  yhr = modelr.predict(x_valid)
  yhlr = modellr.predict(x_valid)
  yht = modelt.predict(x_valid)
  accs_r.append(modelr.evaluate_acc(yhr, y_valid))
  accs_lr.append(modellr.evaluate_acc(yhlr, y_valid))
  accs_t.append(modelt.evaluate_acc(yht, y_valid))


fig = plt.figure()
plt.plot(learning_rates, accs_r, marker= '.', color="blue", alpha=.998, label="Relu")
plt.plot(learning_rates, accs_lr, marker= '.', color="red", alpha=.998, label="Leaky Relu")
plt.plot(learning_rates, accs_t, marker= '.', color="green", alpha=.998, label = "Tanh")
plt.xlabel('Learning Rate')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy as a function of learning rate')
plt.legend()
plt.show()

"""### Test Accuracy of Tanh and Leaky Relu"""

# Define Models

layers_leaky = [Layer(I=x_train.shape[1]), Layer(layer_type='LR', gamma=0.0001), Layer(), Layer(layer_type='LR', gamma=0.0001),Layer(O=10)]
model_lr = MLP(layers=layers_leaky)
layers_tanh = [Layer(I=x_train.shape[1]), Layer(layer_type='T'), Layer(), Layer(layer_type='T'),Layer(O=10)]
model_t = MLP(layers=layers_tanh)

# Fit the model to the 50 000 sample training set, and predict the test outputs

model_lr.fit(x_train, y_train)
print("Leaky relu model fit done")
model_t.fit(x_train, y_train)
print("Tanh model fit done")

yh_lr = model_lr.predict(x_test)
yh_t = model_t.predict(x_test)

print("Test accuracy for leaky relu:" + str(model_lr.evaluate_acc(yh_lr, y_test)) + "\nTest accuracy for tanh :" + str(model_t.evaluate_acc(yh_t, y_test)))

"""###L2 Regularization"""

l2_regs = [0.000001, 0.00001, 0.0001]
accuracies = []

for i in range(len(l2_regs)):
  layers_relu = [Layer(I=x_train.shape[1]), Layer(layer_type='R'), Layer(), Layer(layer_type='R'),Layer(O=10)]
  model = MLP(layers=layers_relu, l2_reg = l2_regs[i])
  model.fit(x_train1, y_train1)
  yh = model.predict(x_valid)
  acc = model.evaluate_acc(yh, y_valid)
  accuracies.append(acc)


fig = plt.figure()
plt.plot(l2_regs, accuracies, marker= '.', alpha=.998)
plt.xlabel('L2 Regularization Strength')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy as a function of L2 Regularization Strength')
plt.show()

"""### L1 Regularization"""

l1_regs = [0.001, 0.01, 0.1]
accuracies = []

for i in range(len(l2_regs)):
  layers_relu = [Layer(I=x_train.shape[1]), Layer(layer_type='R'), Layer(), Layer(layer_type='R'),Layer(O=10)]
  model = MLP(layers=layers_relu)
  model.fit(x_train1, y_train1)
  yh = model.predict(x_valid)
  acc = model.evaluate_acc(yh, y_valid)
  accuracies.append(acc)


fig = plt.figure()
plt.plot(l2_regs, accuracies, marker= '.', alpha=.998)
plt.xlabel('L2 Regularization Strength')
plt.ylabel('Validation Accuracy')
plt.title('Validation Accuracy as a function of L2 Regularization Strength')
plt.show()

"""###Unnormalized Data"""

layers_relu1 = [Layer(I=x_train.shape[1]), Layer(layer_type='R'), Layer(), Layer(layer_type='R'),Layer(O=10)]
modelrelu1 = MLP(layers=layers_relu)

modelrelu1.fit(x_nonorm_train, y_train)

yh6 = modelrelu1.predict(x_nonorm_test)
print("Test Accuracy of unnormalized data:" + str(modelrelu1.evaluate_acc(yh6, y_test)))

"""###Training Curves: Relu"""

layers_relu = [Layer(I=x_train.shape[1]), Layer(layer_type='R'), Layer(), Layer(layer_type='R'),Layer(O=10)]
model = MLP(layers=layers_relu, epochs=50)
model.fit(x_train, y_train)

losses = model.losses


fig = plt.figure()
plt.plot(losses, marker= '.', alpha=.998)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Training curve - Two hidden layers, Relu')
plt.show()

yh = model.predict(x_train)
acc = model.evaluate_acc(yh, y_train)
print(acc)