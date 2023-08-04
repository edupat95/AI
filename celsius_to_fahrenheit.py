import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt

#funcion para convertir de celcius a fahrenheit
def celcius_to_fahrenheit(x):
    return (x * 1.8) + 32

#crear una funcion que cree un conjunto de valores celsius y su resultado en fahrenheit en un arreglo
celsius = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)
fahrenheit = np.array([-40,  14, 32, 46, 59, 72, 100], dtype=float)

#crear una capa de una neurona. Las capas densas son las que tienen conexiones desde cada neurona a todas las de las siguiente capa. En este caso solo tenemos 2 neuronas en la capa
capa = tf.keras.layers.Dense(units=1, input_shape=[1]) #la capa de salida solo tiene una neurona units = 1 y la capa de entrada solo tiene una neurona input_shape = [1]

#crear un modelo con la capa anterior. Hay varios tipos de modelos, el sequiencial es el mas sencillo
modelo = tf.keras.Sequential([capa]) #el modelo solo tiene una capa

#compilar el modelo con un optimizador y una funcion de perdida. El optimizador es el que ajusta los pesos de las neuronas para que el resultado sea el esperado. La funcion de perdida es la que mide que tan lejos esta el resultado de lo esperado
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1), #Adam es un optimizador que ajusta la tasa de aprendizaje
    loss='mean_squared_error' #funcion de perdida. Basicamente concidera que una poca cantidad de errores grandes es peor que muchos errores pequeños.
)

#entrenar el modelo con los datos de entrada y salida esperados. epochs es la cantidad de veces que se va a entrenar el modelo
print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1500, verbose=False) # le damos 1000 vueltas 
print("Modelo entrenado!")

#graficar el historial de perdida
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(historial.history["loss"])
#plt.show()  # Mostrar el gráfico

#evaluar el modelo con un valor de entrada
print("Hagamos una prediccion!")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + " fahrenheit!")

#mostrar los pesos de la neurona
print("Variables internas del modelo")
print(capa.get_weights()) #la primera es el peso y la segunda es el bias
