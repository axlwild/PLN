"""
Cervantes Gonzalez Victor Axel
UNAM.
Facultad de Ingeniería.

"""

from collections import Counter
import numpy as np
encoding='utf-8-sig'

archCorpus = "corpus_HMM.txt"
corpus = open(archCorpus,"r")
corpusText = corpus.read().lower().split()

#Sigma
palabras = []
etiquetas = []
contador = Counter()

#print(corpusText)

#Separando palabras de etiquetas
for i in range(0,len(corpusText)-1,2):
	palabras.append(corpusText[i])
	etiquetas.append(corpusText[i+1])

#Para contar más rápido, hacemos un counter de las etiquetas y las veces que está la etiqueta.
contador.update(etiquetas)
#Tamaño del alfabeto
n = len(contador)
lamb = 1

#Se asigna un número a cada etiqueta, se ordenan alfabéticamente.
etiquetasID = list(contador)
etiquetasID.sort()
#Pero necesito la etiqueta como llave:
etiquetasDic = {}
for idx, et in enumerate(etiquetasID):
	etiquetasDic[et] = idx

#A: Matriz de transiciones
A = np.zeros([len(etiquetasID),len(etiquetasID)])

#Llenando la matriz de probabilidades de transiciones
for i in range(1,len(etiquetas)-1):
	A[ etiquetasDic[etiquetas[i]] ][ etiquetasDic[etiquetas[i-1]] ] += 1


#Aplicando smothing laplaciano:
A += lamb
for idx,et in enumerate(etiquetasID):
	for idy,et1 in enumerate(etiquetasID):
		A[idx][idy] *= 1/(contador[et1] + lamb*n)

PI = np.zeros(len(etiquetasID))
#Considerando que solo hay un inicial
PI[etiquetasDic[etiquetas[0]]] = 1
PI += lamb
for idx,et in enumerate(etiquetasID):
	PI[idx] *= 1/(n+lamb)
"""
print("Modelo de lenguaje:")
print("A:")
print(A)
print("PI:")
print(PI)
print("Sigma:")
print(etiquetasID)
"""







		
















