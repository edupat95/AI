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
b1 = np.random.rand(1, N_hidden)  # Sesgos para la capa oculta

W2 = np.random.rand(N_hidden, N_output) # Matriz de pesos iniciales (w31, w32)
b2 = np.random.rand(1, N_output)  # Sesgos para la capa de salida

print ("Pesos iniciales de la capa de entrada a la capa oculta: \n", W1)
print ("Pesos iniciales de la capa oculta a la capa de salida: \n", W2)

# Inicializar listas para almacenar información de seguimiento
loss_history = []
accuracy_history = []
weights_history = []

for epoch in range(10000):
    # Forward Propagation
    Z1 = np.dot(inputs, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    y_hat = sigmoid(Z2)
    
    # Backpropagation
    # Calcular el error
    error = y - y_hat
    
    # Calcular la derivada del error con respecto a la salida
    dZ2 = error * sigmoid(Z2) * (1-sigmoid(Z2))
    
    # Calcular la derivada del error con respecto a la capa oculta
    dZ1 = np.dot(dZ2, W2.T) * sigmoid(Z1) * (1-sigmoid(Z1))
    
    # Calcular pérdida y precisión en cada época
    loss = np.mean(np.square(error))
    accuracy = np.mean(np.round(y_hat) == y)
    
    loss_history.append(loss)
    accuracy_history.append(accuracy)
    weights_history.append(W1.copy())  # Guardar copia de los pesos
    
    # Actualizar pesos
    W2 += np.dot(A1.T, dZ2)
    W1 += np.dot(inputs.T, dZ1)

# Graficar la Pérdida vs. Épocas
plt.figure()
plt.plot(loss_history)
plt.title('Pérdida vs. Épocas')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.show()

# Graficar la Precisión vs. Épocas
plt.figure()
plt.plot(accuracy_history)
plt.title('Precisión vs. Épocas')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.show()

# Graficar Peso vs. Épocas
weights_history = np.array(weights_history)
plt.figure()
for i in range(weights_history.shape[1]):
    plt.plot(weights_history[:, i], label=f'W1[:, {i}]')
plt.title('Evolución de Pesos vs. Épocas')
plt.xlabel('Épocas')
plt.ylabel('Valor de Peso')
plt.legend()
plt.show()
# Predecir entradas
inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
Z1 = np.dot(inputs, W1)
A1 = sigmoid(Z1)
Z2 = np.dot(A1, W2)
y_hat = sigmoid(Z2)

print ("Salida predecida: \n", y_hat)