import numpy as np
import tensorflow as tf
from sklearn.neural_network import MLPClassifier as mlp
from utilities import *
from sklearn.metrics import confusion_matrix
import sys
import os


def personalResults(predY, testY):
	predClass = np.argmax(predY,1)
	C = confusion_matrix(np.argmax(testY,1).ravel(), predClass.ravel(), np.arange(n_classes))
	np.set_printoptions(suppress=True)
	return C

	
# --------******** main *********----------

# Hiperparámetros y rutas de datos
n_classes = 7
ret = 0
#IDs = ['100','101','103','105','106','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'] # all records
IDs = ['200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
allCs={}
outArr = np.zeros((1,8)) # matriz de 8 elementos para la salida
runs = np.random.permutation(np.arange(int(sys.argv[1]), int(sys.argv[2])))*ret#['_1', '_2', '_3', ..., '_50']
out_path = 'finalResults.csv'
os.system('rm ' + out_path)


# *** Construimos Modelos ***
for run in runs:
	print("blending models ...")
	
	# Bucle para todos los pacientes
	for ID in IDs:
		
		# Cargamos los datos de X de dos modelos para entrenamiento
		XB = np.load('../preds/trainB_outs_'+str(run)+'_' + str(ret)+'/' + ID + '_predY.npy')
		XA = np.load('../preds/trainA_outs_'+str(run)+'_' + str(ret)+'/' + ID + '_predY.npy')
		if(np.size(XA,0)!=np.size(XB,0)):
			XB = XB[0:np.size(XA,0),:]
		XB = np.argmax(XB,1)
		XB = np.eye(n_classes)[XB.astype(int)]
		XA = np.argmax(XA,1)
		XA = np.eye(n_classes)[XA.astype(int)]
		
		# Se hace el entrenamiento con ambos modelos
		trainX = np.concatenate((XB,XA),1)
		trainY = np.load('../Data/label/' + ID + '/trainYcat.npy')
		
		# Ajustamos un modelo mlp
		clf = mlp(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(80,10), random_state = 1)
		clf.fit(trainX, trainY)
	
		# Cargamos los datos de X de dos modelos para prueba
		testX_B = np.load('../preds/testB_outs_'+str(run)+'_' + str(ret)+'/' + ID + '_predY.npy')
		testX_A = np.load('../preds/testA_outs_'+str(run)+'_' + str(ret)+'/' + ID + '_predY.npy')
		testX = np.concatenate((testX_B,testX_A),1)
		
		# Obtenemos los resultados del modelo
		predY = clf.predict(testX)
		
		# Calculamos las precisiones
		testY = np.load('../Data/label/' + ID + '/testYcat.npy')
		C = personalResults(predY, testY)
		allCs[ID] = C
		
	outArr += calc_tables(allCs, n_classes)
outArr /= np.size(runs)

# Guardamos los resultados en un archivo csv
print("final results:\n", outArr)
with open(out_path, 'ab') as out_handle:
	np.savetxt(out_handle, outArr, delimiter=',')
