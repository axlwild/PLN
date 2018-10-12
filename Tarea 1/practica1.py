"""
Cervantes Gonzalez Victor Axel
UNAM.
Facultad de Ingeniería.
"""
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from collections import Counter
from matplotlib import pyplot as plt



estemizador = SnowballStemmer('spanish')
contador = Counter()
archivo = open("G1.txt", "r+")
arch_frec = open("frecuencias.txt","w+")
cadena = archivo.read();
token = []
frecuencia = []

#Paso 1: Limpiar el corpus
str_clean = ""
signos = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_{|}~`¡¿")
for letra in cadena:
	if letra not in signos:
		str_clean += str(letra)

#Paso 2: Aplicar un algoritmo de stemming a los tokens limpios.
texto_stemizado = [estemizador.stem(i) for i in word_tokenize(str_clean)]
#print(texto_stemizado)


#Paso 3: Obtener las frecuencias de los tipos en el corpus.
contador.update(texto_stemizado)

#Paso 4: Ordenar por el rango estadístico de mayor a menos
contador = list(contador.most_common(len(contador)))
for elemento in contador:
	token += [elemento[0]]
	frecuencia += [elemento[1]]
#Pasamos las frecuencias a un archivo
for i in range(len(token)):
	arch_frec.write("{}\t\t:\t\t{}\n".format(frecuencia[i],token[i]))



#Paso 5: Graficar diagrama de dispersión rango-frecuencia en escala logarítmica
rango = range(len(frecuencia))
ploteo = plt.plot(rango, frecuencia,"ro")
plt.setp(ploteo,markersize=.5)
plt.xscale('log')
#plt.axis('off')
plt.show()

archivo.close()
arch_frec.close()
