from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Nivel de advertencia
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # usa GPU 0
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import sys
from math import ceil


def buildGraph():
	global x1
	global x2
	global y
	global weights
	global biases
	global cost
	global optimizer
	global correct_pred
	global accuracy
	global pred
	global init
	global saver
	
	with tf.device('/gpu:0'):

		# reset Graph
		tf.reset_default_graph()

		# tf Graph input
		x1 = tf.placeholder("float", [None, n_steps1, n_input1], name='x1')
		x2 = tf.placeholder("float", [None, n_steps2, n_input2], name='x2')
		y = tf.placeholder("float", [None, n_classes], name='y')

		# Preparamos la forma de los datos para que coincida con los requisitos de la función `rnn`
		# Forma de entrada de datos actual: (batch_size, n_steps, n_input)
		# Forma requerida: 'n_steps' lista de tensores de forma (batch_size, n_input)
		# Permutación del batch_size y n_steps
		x11 = tf.transpose(x1, [1, 0, 2])
		x22 = tf.transpose(x2, [1, 0, 2])
		# Remodelamos a (n_steps*batch_size, n_input)
		x11 = tf.reshape(x11, [-1, n_input1])
		x22 = tf.reshape(x22, [-1, n_input2])
		# Separamos para obtener una lista de 'n_steps' tensores de forma (batch_size, n_input)  
		x11 = tf.split(x11, n_steps1, 0)
		x22 = tf.split(x22, n_steps2, 0)

		# Definimos los pesos
		weights = tf.Variable(tf.random_normal([n_hidden1+n_hidden2, n_classes]))
		biases = tf.Variable(tf.random_normal([n_classes]))

		# Definimos celdas lstm con tensorflow
		lstm_cell1 = rnn.LSTMCell(n_hidden1, use_peepholes=False)
		lstm_cell2 = rnn.LSTMCell(n_hidden2, use_peepholes=False)

		# Obtenemos salidas de células lstm
		with tf.variable_scope('rnn1'):
			outputs1, states1 = rnn.static_rnn(lstm_cell1, x11, dtype=tf.float32)
		with tf.variable_scope('rnn2'):
			outputs2, states2 = rnn.static_rnn(lstm_cell2, x22, dtype=tf.float32)
	
		# Activación lineal, utilizando la última salida del bucle interno rnn
		tmp1 = tf.concat(values=[outputs1[-1], outputs2[-1]], axis=1)
		tmp1 = tf.matmul(tmp1, weights)
		pred = tf.add(tmp1, biases, name='pred')

		# Definimos la pérdida y optimizador
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

		# Evaluamos el modelo
		correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')

		# Inicializamos las variables
		init = tf.global_variables_initializer()
	
		# Escribimos el modelo
		#writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

		# Opción 'Saver' para guardar y restaurar todas las variables
		saver = tf.train.Saver()
		
	return


def runModel(ID):
	global sess
	global predY
	global trainX1
	global trainX2
	global trainYcat
	
	# Establecemos las configuraciones
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True # Utilizamos la memoria de las GPU de forma más eficiente

	# Launch the graph
	with tf.Session(config=config) as sess:
		sess.run(init)
		
		for curr_epoch in range(num_epochs):
			for batch in range(num_batches_per_epoch):
			
				# Preparamos el lote requerido
				batchRange = range(batch*batch_size, min((batch+1)*batch_size,trainX1.shape[0]))
				batch_x1 = trainX1[batchRange,:,:]
				batch_x2 = trainX2[batchRange,:,:]
				batch_y = trainYcat[batchRange,:]
				
				# Cambiamos la forma de los datos para obtener una secuencia de elementos necesarios
				batch_x1 = batch_x1.reshape((batch_x1.shape[0], n_steps1, n_input1))
				batch_x2 = batch_x2.reshape((batch_x2.shape[0], n_steps2, n_input2))
				
				# Ejecutamos la operación de optimización (backprop)
				sess.run(optimizer, feed_dict={x1: batch_x1, y: batch_y, x2:batch_x2})
				
				# Calculamos la precisión del último lote de época y la pérdida de lote
				acc, loss = sess.run([accuracy, cost], feed_dict={x1: batch_x1, y: batch_y, x2:batch_x2})
			
			#losses.append(loss)
			#acces.append(acc)
		
		# Guardamos la predicción de los datos del entrenamiento
		predY = sess.run(pred, feed_dict={x1: trainX1, y: trainYcat, x2:trainX2})
		directory = os.path.dirname(predsPath)
		try:
			os.stat(directory)
		except:
			os.mkdir(directory)
		np.save( predsPath + ID + '_predY.npy', predY)
		
		# Guardamos los pesos del modelo en el disco
		directory = os.path.dirname(model_path)
		try:
			os.stat(directory)
		except:
			os.mkdir(directory)
		model_path_ID = model_path + ID + '/'
		directory = os.path.dirname(model_path_ID)
		try:
			os.stat(directory)
		except:
			os.mkdir(directory)
		del directory
		save_path = saver.save(sess, model_path_ID+'model.ckpt')
		print("ModelA of %s was saved" % str(ID+'_'+str(run)))
	
	return


def loadTrainVars(ID):
	global num_batches_per_epoch
	global trainX1
	global trainX2
	global trainYcat
	global n_input1
	global n_input2

	# Cargamos las variables de entrenamiento
	vars_path_ID = vars_path + ID + '/'
	trainX = np.load(vars_path_ID+'trainX.npy')
	trainYcat = np.load(vars_path_ID+'trainYcat.npy')
	
	# Enmascaramos algunas partes para hacer la señal deseada
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
	trainX1 = trainX[:,:,mask1]
	trainX2 = trainX[:,:,mask2]
	
	
	num_examples = trainX.shape[0]
	num_batches_per_epoch = ceil(num_examples/batch_size)
	n_input1 = (trainX1.shape[2]//n_steps1)
	n_input2 = (trainX2.shape[2]//n_steps2)
	trainX1 = trainX1.reshape((trainX1.shape[0], n_steps1, n_input1))
	trainX2 = trainX2.reshape((trainX2.shape[0], n_steps2, n_input2))
	
	return



# --------******** main *********----------

# Hiperparámetros
n_classes = 7
n_steps = 10
n_hidden = 30
num_epochs = 100
num_layers = 1
batch_size = 100
learning_rate = 1e-3
ret = 0
n_steps1 = n_steps
n_steps2 = n_steps
n_hidden1 = n_hidden
n_hidden2 = n_hidden

# Rutas de datos
vars_path = '../Data/AA/'
models_path = '../models/'
logs_path = './Logs/'
IDs = ['100','101','103','105','106','108','109','111','112','113','114','115','116','117','118','119','121','122','123','124','200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234'] # all records
#IDs = ['200', '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '219', '220', '221', '222', '223', '228', '230', '231', '232', '233', '234']
runs = np.random.permutation(np.arange(int(sys.argv[1]), int(sys.argv[2])))#ret#['_1', '_2', '_3', ..., '_50']

# ***Construimos los modelos***
# Bucle para todos los pacientes
for ID in IDs:
	loadTrainVars(ID) #Cargamos las variables

	for run in runs:
		# Ruta de almacenamiento de modelos y sus resultados
		model_path = models_path + 'modelsA_' + str(run) + '_' + str(ret) + '/'
		predsPath = '../preds/trainA_outs_' + str(run) + '_' + str(ret) + '/'
		buildGraph() #make the graph
		runModel(ID) #Entrenamos el modelo con los datos de entrenamiento
