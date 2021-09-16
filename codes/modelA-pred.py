from __future__ import print_function
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Nivel de advertencia
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # usa GPU 0
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from utilities import *
from sklearn.metrics import confusion_matrix


def runModel(ID):
	global sess
	global predY
	global testAcc
	
	# Establecemos las configuraciones
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:
	
		# Restauramos los pesos del modelo del modelo previamente guardado
		model_path_ID = model_path + ID + '/'
		saver = tf.train.import_meta_graph(model_path_ID+'model.ckpt.meta')
		saver.restore(sess,tf.train.latest_checkpoint(model_path_ID))
		print("Model of %s is restored" % str(ID+'A_'+str(run)))

		# Restauramos las variables del modelo del modelo previamente guardado
		graph = tf.get_default_graph()
		accuracy = graph.get_tensor_by_name("Accuracy:0")
		x1 = graph.get_tensor_by_name("x1:0")
		x2 = graph.get_tensor_by_name("x2:0")
		y = graph.get_tensor_by_name("y:0")
		pred = graph.get_tensor_by_name("pred:0")

		# Ejecutamos el modelo con datos de prueba
		testAcc, predY = sess.run([accuracy, pred], feed_dict={x1: testX1, y: testYcat, x2: testX2})
		
		# Guardamos las predicciones
		directory = os.path.dirname(predsPath)
		try:
			os.stat(directory)
		except:
			os.mkdir(directory)
		np.save( predsPath + ID + '_predY.npy', predY)
	
	return
	

def personalResults(): # Esta función calcula la matriz de confusión para cada ID
	predClass = np.argmax(predY,1)
	C = confusion_matrix(testY.ravel(), predClass.ravel(), np.arange(n_classes))
	np.set_printoptions(suppress=True)
	return C


def loadTestVars(ID):
	global testX1
	global testX2
	global testY
	global testYcat

	# Cargamos variables de prueba
	vars_path_ID = vars_path + ID + '/'
	testX = np.load(vars_path_ID+'testX.npy')
	testY = np.load(vars_path_ID+'testY.npy')
	testYcat = np.load(vars_path_ID+'testYcat.npy')
	
	# enmascaramos algunas partes para hacer la señal deseada
	mask1 = np.zeros((780,), dtype=bool)
	mask2 = np.zeros((780,), dtype=bool)
	mask1[0:4]=1
	mask1[0+390:4+390]=1
	mask1[4:4+250+1]=1
	mask1[4+390:4+390+250+1]=1
	mask2[0:4]=1
	mask2[0+390:4+390]=1
	mask2[4+250+1:4+250+1+135]=1
	mask2[4+250]=1
	mask2[4+390+250+1:4+390+250+1+135]=1
	mask2[4+390+250]=1
	testX1 = testX[:,:,mask1]
	testX2 = testX[:,:,mask2]
	
	
	num_examples = testX.shape[0]
	n_input1 = (testX1.shape[2]//n_steps1)
	n_input2 = (testX2.shape[2]//n_steps2)
	testX1 = testX1.reshape((num_examples, n_steps1, n_input1))
	testX2 = testX2.reshape((num_examples, n_steps2, n_input2))
	
	return
	

	
# --------******** main *********----------

# Hiperparámetros
n_classes = 7
n_steps = 10
n_hidden = 30
ret=0
n_steps1 = n_steps
n_steps2 = n_steps

# Rutas de datos
vars_path = '../Data/AA/'
models_path = '../models/'
res_path = models_path + 'resA.txt'
os.system('rm ' + res_path)
IDs = ['100','101','103','105','106','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'] # all records
#IDs = ['200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
runs = np.random.permutation(np.arange(int(sys.argv[1]), int(sys.argv[2])))#ret#['_1', '_2', '_3', ..., '_50']
outArr = np.zeros((1,8)) # array de 8 elementos para la salida

# ***Cargamos los modelos***
for run in runs:

	# Ruta de almacenamiento de modelos y sus resultados
	model_path = models_path + 'modelsA_' + str(run) + '_' + str(ret) + '/'
	predsPath = '../preds/testA_outs_' + str(run) + '_' + str(ret) + '/'
	allCs = {}
		
	# Bucle para todos los pacientes
	for ID in IDs:
		loadTestVars(ID) #Cargamos las variables	
		tf.reset_default_graph() #reset Graph
		runModel(ID) #Ejecutamos el modelo completo con nuevos datos de prueba
		C = personalResults() #Vemos los resultados
		allCs[ID] = C #Almacenamos matrices de confusión
	
	outArr += calc_tables(allCs, n_classes)

outArr /= np.size(runs)
