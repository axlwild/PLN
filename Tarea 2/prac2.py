"""
Cervantes Gonzalez Victor Axel
UNAM.
Facultad de Ingeniería.

Notas:
El algoritmo solo funciona con palabras dentro del corpus.
Se consideró el alfabeto como las etiquetas.
Se desarrolló este programa para python3.
Se necesita numpy para este algoritmo y pickle para 
guardarlo en este formato.
"""


from pickle import dump, dumps, load, loads
from collections import Counter
import numpy as np
encoding='utf-8-sig'
np.set_printoptions(threshold=np.inf)

#################################Definición del Algoritmo de Viterbi#######################################
def viterbi(obs, etiquetasID,A,B, etiquetasDic, palID, palDic, PI):
	#La frase a analizar.
	print("frase a analizar: "+obs)
	palabra = obs.split()
	d = np.zeros([len(palabra)])
	phi = np.zeros([len(etiquetasID),len(palabra)])
	#Existen n deltas, donde n es el número de emisiones
	delta = np.zeros([len(etiquetasID),len(palabra)])
	#Inicialización
	auxDelta = delta.transpose()
	auxDelta[0] = B[palDic[palabra[0]]]*PI.transpose()
	delta = auxDelta.transpose()		
	#Inducción:
	for idj, j in enumerate(palabra[1:]):
		for idx, i in enumerate(etiquetasID):
			delta[idx][idj+1] = max(B[palDic[j]][idx]*A[idx]*delta.transpose()[idj]) 
			phi[idx][idj] = np.argmax(B[palDic[j]][idx]*A[idx]*delta.transpose()[idj]) 
	#Retroceso:
	d[-1] = np.argmax(delta.transpose()[-1])
	for i in reversed(range(len(palabra)-1)):
		#print(i)
		d[i] = phi.transpose()[i][int(d[i+1])]
	#Finalización:
	for idx,i in enumerate(d):
		print(etiquetasID[int(i)],end="\t")
		print(palabra[idx])
	print()


def prueba(cadena):
	viterbi(cadena,etiquetasID,A,B,etiquetasDic,palID,palDic,PI)

############################################INICIO########################################################
archCorpus = "corpus_HMM.txt"
corpus = open(archCorpus,"r")
corpusText = corpus.read().lower().split()

#Para separar etiquedas de etiquetas, y contar cuándos elementos repetidos hay
palabras = []
etiquetas = []
contPal = Counter()
contador = Counter()

#Diccionario palabras, etiquetas.
palabrasEtiquetas = {}
#Separando palabras de etiquetas
for i in range(0,len(corpusText)-1,2):
	palabras.append(corpusText[i])
	etiquetas.append(corpusText[i+1])
	if etiquetas[-1] in palabrasEtiquetas:
		palabrasEtiquetas[etiquetas[-1]].update({palabras[-1]})
	else:
		palabrasEtiquetas[etiquetas[-1]] = Counter({palabras[-1]})

#Para contar más rápido, hacemos un counter de las etiquetas y las veces que está la etiqueta.
contador.update(etiquetas)
#Tamaño del alfabeto
n = len(contador)
lamb = 1
#Creamos contador de palabras, con el fin de obtener las palabras únicas.
contPal.update(palabras)
palID = list(contPal)
palID.sort()
#print(palID)
palDic = {}
for idx, pal in enumerate(palID):
	palDic[pal] = idx

#Se asigna un número a cada etiqueta, se ordenan alfabéticamente.
etiquetasID = list(contador)
etiquetasID.sort()
#Pero necesito la etiqueta como llave, para agilizar el algoritmo.
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

PI = np.zeros([len(etiquetasID),1])
#Considerando que solo hay un inicial
PI[etiquetasDic[etiquetas[0]]][0] = 1
PI += lamb
for idx,et in enumerate(etiquetasID):
	PI[idx][0] *= 1/(n+n*lamb)


B = np.zeros([len(palID),len(etiquetasID)])
#Iniciando a la vez el smoothing 
B += lamb
etsyPalTot = len(palID)+len(etiquetasID)
for idi,i in enumerate(palID):
	for idj,j in enumerate(etiquetasID):
		B[idi][idj] = (B[idi][idj] + palabrasEtiquetas[j][i])/(contador[j]+contPal[i]+etsyPalTot*lamb)


"""
print("Modelo de lenguaje:")
print("\nA:")
print(A)
print("\nPI:")
print(PI)
print("\nSigma:")
print(etiquetasID)
print("\nB:")
print(B)
"""
HMM = [etiquetasID,A,PI,B]
HMM_Pickle = dumps(HMM)

#Probando el algoritmo

prueba("en la ciudad de sarnath distrito")
prueba("el primer despertar de una yuga")
prueba("la cosmología budista hace esta distinción a el afirmar que	únicamente los humanos pueden lograr el estado de buda")


		
	













