from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Niveles de advertencia
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # usa GPU 0
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import sys
from math import ceil


def buildGraph():
	global x
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
		x = tf.placeholder("float", [None, n_steps, n_input], name='x')
		y = tf.placeholder("float", [None, n_classes], name='y')

		# Preparamos la forma de los datos para que coincida con los requisitos de la función `rnn`
		# Forma de entrada de datos actual: (batch_size, n_steps, n_input)
		# Forma requerida: 'n_steps' tensors list of shape (batch_size, n_input)
		# Permutación de batch_size y n_steps
		x1 = tf.transpose(x, [1, 0, 2])
		# Remodelamos a (n_steps*batch_size, n_input)
		x1 = tf.reshape(x1, [-1, n_input])
		# Separamos para obtener una lista de 'n_steps' tensores de forma (batch_size, n_input)
		x1 = tf.split(x1, n_steps, 0)

		# Definimos los pesos
		weights = tf.Variable(tf.random_normal([n_hidden, n_classes]))
		biases = tf.Variable(tf.random_normal([n_classes]))

		# Definimos una celda lstm con tensorflow
		lstm_cell = rnn.LSTMCell(n_hidden, use_peepholes=False)

		# Obtenemos salida de celda lstm
		outputs, states = rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)
	
		# Activación lineal, utilizando la última salida del bucle interno rnn
		tmp1 = tf.matmul(outputs[-1], weights)
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
	global acc
	global loss
	global losses
	global acces
	global testAccs
	global trainX
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
				batch_x = trainX[batch*batch_size:(batch+1)*batch_size,:,:]
				batch_y = trainYcat[batch*batch_size:(batch+1)*batch_size,:] 
				
				# Cambiamos la forma de los datos para obtener 28 segundos de 28 elementos
				batch_x = batch_x.reshape((batch_size, n_steps, n_input))
				
				# Ejecutamos la operación de optimización (backprop)
				sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
				
				# Calculamos la precisión del último lote de época y la pérdida de lote
				acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})
			
			#losses.append(loss)
			#acces.append(acc)
			
		# Guardamos los datos de entrenamiento
		predY = sess.run(pred, feed_dict={x: trainX, y: trainYcat})
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
		print("ModelB of %s was saved" % str(ID+'_'+str(run)))
	
	return


def loadTrainVars(ID):
	global num_batches_per_epoch
	global trainX
	global trainYcat
	global n_input

	# Cargamos las variables de entrenamiento
	vars_path_ID = vars_path + ID + '/'
	trainX = np.load(vars_path_ID+'trainX.npy')
	trainYcat = np.load(vars_path_ID+'trainYcat.npy')
	
	
	num_examples = trainX.shape[0]
	num_batches_per_epoch = int(num_examples//batch_size)
	n_input = (trainX.shape[2]//n_steps)
	trainX = trainX.reshape((trainX.shape[0], n_steps, n_input))
	
	return

	

# ----------******main******---------

# Hiperparámetros
n_classes = 7
num_epochs = 100
num_layers = 1
batch_size = 100
learning_rate = 1e-3
ret = 0
n_hidden = 50
n_steps = 5

# Rutas de datos
vars_path = '../Data/BB/'
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
		model_path = models_path + 'modelsB_' + str(run) + '_' + str(ret) + '/'
		predsPath = '../preds/trainB_outs_' + str(run) + '_' + str(ret) + '/'
		buildGraph() #make the graph
		runModel(ID) #Entrenamos el modelo con los datos de entrenamiento
