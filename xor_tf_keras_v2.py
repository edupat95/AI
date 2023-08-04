import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Define el numero de neuronas en la capa de entrada, oculta y salida
N_input = 2
N_hidden = 4 
#N_hidden2 = 2
N_output = 1

# Inicializamos los pesos
W1 = tf.Variable(tf.random.normal([N_input, N_hidden])) # 2 * 4
W2 = tf.Variable(tf.random.normal([N_hidden, N_output]))

# Definimos la funcionde activacion
def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

# Definimos el modelo
model = Sequential()
model.add(Dense(N_hidden, input_dim=N_input, activation=sigmoid)) #Capa oculta, 2 neuronas, 2 entradas, funcion de activacion sigmoid. Densa es una capa de neuronas conectadas a todas las neuronas de la capa anterior
model.add(Dense(N_output, activation=sigmoid)) #Capa de salida, 1 neurona, 2 entradas, funcion de activacion sigmoid

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the data
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Lists to store weights for plotting
W1_history = []
W2_history = []

# Custom callback to record weights after each epoch
class WeightHistory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        W1_history.append(self.model.layers[0].get_weights()[0])
        W2_history.append(self.model.layers[1].get_weights()[0])

# Train the model and use the custom callback
history = model.fit(inputs, y, epochs=5000, callbacks=[WeightHistory()])

# Predict the inputs
y_hat = model.predict(inputs)

# Print the predicted outputs
print(y_hat)

# Make a prediction
new_inputs = np.array([[1, 1], [0, 1], [0, 0], [1, 0]])
prediction = model.predict(new_inputs)

print("Predict de new_inputs [1, 1], [0, 1], [0, 0], [1, 0]-> ", prediction)

# Convert weight history lists to numpy arrays
W1_history = np.array(W1_history)
W2_history = np.array(W2_history)

# Plot the weights evolution during training
plt.figure(figsize=(10, 6))
plt.plot(W1_history[:, 0, 0], label='W1[0, 0]')
plt.plot(W1_history[:, 0, 1], label='W1[0, 1]')
plt.plot(W1_history[:, 0, 2], label='W1[0, 2]')
plt.plot(W1_history[:, 0, 3], label='W1[0, 3]')
plt.plot(W1_history[:, 1, 0], label='W1[1, 0]')
plt.plot(W1_history[:, 1, 1], label='W1[1, 1]')
plt.plot(W1_history[:, 1, 2], label='W1[1, 2]')
plt.plot(W1_history[:, 1, 3], label='W1[1, 3]')
#Plot the weights evolution of the second layer (capa de salida)

plt.plot(W2_history[:, 0, 0], label='W2[0, 0]')
plt.plot(W2_history[:, 1, 0], label='W2[1, 0]')
plt.plot(W2_history[:, 2, 0], label='W2[2, 0]')
plt.plot(W2_history[:, 3, 0], label='W2[3, 0]')

print("W1 -> ", W1) # Mostramos el valor final que tienen los pesos correspondiente a la capa de entrada a la capa oculta (2 x 2)
print("W2 -> ", W2) # Mostramos el valor final que tienen los pesos correspondiente a la capa oculta a la capa de salida (2 x 1)

plt.xlabel('Epoch')
plt.ylabel('Weight Value')
plt.title('Evolution of Weights in First Layer')
plt.legend()
plt.grid()
plt.show()


# Plot the epoch_loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss during Training')
plt.grid()
plt.show()

# Plot the epoch_accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy during Training')
plt.grid()
plt.show()




