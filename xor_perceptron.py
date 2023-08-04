import numpy as np
import matplotlib.pyplot as plt

np.random.seed(37)

inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

#definir funcion de activacion 
def sigmoid(x):
    return 1/(1+np.exp(-x))


#Definir numero de entradas
N_input = 2
N_hidden = 2
N_output = 1

#Inicializar pesos
W1 = np.random.rand(N_input, N_hidden) # Matriz de pesos iniciales (w11, w12, w21, w22)
W2 = np.random.rand(N_hidden, N_output) # Matriz de pesos iniciales (w31, w32)

print ("Pesos iniciales de la capa de entrada a la capa oculta: \n", W1)
print ("Pesos iniciales de la capa oculta a la capa de salida: \n", W2)


for epoch in range(10000):
    # Forward Propagation
    Z1 = np.dot(inputs, W1) #Aplicamos la sumatoria de los pesos por las entradas.
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    y_hat = sigmoid(Z2)
    
    # Backpropagation
    # Calcular el error
    error = y - y_hat
    
    # Calcular la derivada del error con respecto a la salida
    dZ2 = error * sigmoid(Z2) * (1-sigmoid(Z2))
    
    # Calcular la derivada del error con respecto a la capa oculta
    dZ1 = np.dot(dZ2, W2.T) * sigmoid(Z1) * (1-sigmoid(Z1))
    
    # Actualizar pesos
    W2 += np.dot(A1.T, dZ2)
    W1 += np.dot(inputs.T, dZ1)

# Predecir entradas
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
Z1 = np.dot(inputs, W1)
A1 = sigmoid(Z1)
Z2 = np.dot(A1, W2)
y_hat = sigmoid(Z2)

print ("Salida predecida: \n", y_hat)