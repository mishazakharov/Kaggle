import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from functools import partial


data = pd.read_csv('train.csv')

X = data.drop('Survived',axis=1)
y = data['Survived']

def check_numerical(a):
	'''checking whether input is numerical or not'''
	return isinstance(a,np.int64) or isinstance(a,np.float64)

def merge_vectors(numericals):
	''' Сливает все векторы из numericals в один DataFrame. '''
	result = numericals[0]
	for i in range(1,len(numericals)):
		result = pd.merge(result,numericals[i],left_index=True,
									right_index=True,how='left')
	return result

def prepare_dataset(columns,train):
	''' Подгатавливает датасет к обучающему алгоритму! '''

	# Список для хранения числовых форматированных атрибутов
	numericals = []

	for attribute in columns:
	# Если аттрибут числовой
		if check_numerical(train[attribute][0]):
			if int(train[attribute].count()) == 1460:
				numericals.append(train[attribute])
				continue
			imputer = Imputer(strategy='median')
			imputer.fit(train[attribute].values.reshape(-1,1))
			# Колонна с текущим атрибутом
			X = imputer.transform(train[attribute].values.reshape(-1,1))
			X = pd.DataFrame(X,columns=[attribute])
			numericals.append(X)
	# Если аттрибут категориальный 
		else:
			X_encoded,categories = train[attribute].factorize()
			# Перевод в pandas.DataFrame
			x = pd.DataFrame(X_encoded.reshape(-1,1),columns=[attribute])
			numericals.append(x)

	return numericals

columns = X.columns

# Подготовка датасета для алгоритма! 
numericals = prepare_dataset(columns,X)
X_ready = merge_vectors(numericals)
X_ready = X_ready.drop('PassengerId',axis=1)
y_ready = y


# Для обучения(разделение датасетов)
#X_train,X_test,y_train,y_test = train_test_split(X_ready,y,test_size=0.07)

# Строим нейронную сеть low-level API tensorflow!
# Параметры сети
n_inputs = 10
n_hidden1 = 400
n_hidden2 = 300
n_outputs = 2
n_epochs = 631

X = tf.placeholder(tf.float32,shape=(None,10),name='X')
training = tf.placeholder_with_default(True,shape=(),name='training')
y = tf.placeholder(tf.int32,shape=(None),name='y')

# Гиперпараметр
dropout_rate = 0.5
X_drop = tf.layers.dropout(X,dropout_rate,training=training)

with tf.name_scope('dnn'):
	# Пакетная нормализация на все слои! 
	my_batch_norm_layer = partial(tf.layers.batch_normalization,
									training=training,momentum=0.20)
	# Регуляризация l1 плохо работает!!!
	# Также как и ОТКЛЮЧЕНИЕ! 
	hidden1 = tf.layers.dense(X_drop,n_hidden1,name='hidden1')
	bn1 = my_batch_norm_layer(hidden1)
	bn1_act = tf.nn.relu(bn1)
	hidden2 = tf.layers.dense(bn1_act,n_hidden2,name='hidden2')
	bn2 = my_batch_norm_layer(hidden2)
	bn2_act = tf.nn.relu(bn2)
	hidden3 =tf.layers.dense(bn2_act,175,name='hidden3')
	bn3 = my_batch_norm_layer(hidden3)
	bn3_act = tf.nn.relu(bn3)
	hidden4 = tf.layers.dense(bn3_act,150,name='hidden4')
	bn4 = my_batch_norm_layer(hidden4)
	bn4_act = tf.nn.relu(bn4)
	hidden5 = tf.layers.dense(bn4_act,125,name='hidden5')
	bn5 = my_batch_norm_layer(hidden5)
	bn5_act = tf.nn.relu(bn5)
	hidden6 = tf.layers.dense(bn5_act,1000,name='hidden6')
	bn6 = my_batch_norm_layer(hidden6)
	bn6_act = tf.nn.relu(bn6)
	hidden7 = tf.layers.dense(bn6_act,40,name='hidden7')
	bn7 = my_batch_norm_layer(hidden7)
	bn7_act = tf.nn.relu(bn7)
	hidden8 = tf.layers.dense(bn7_act,50,name='hidden8')
	bn8 = my_batch_norm_layer(hidden8)
	bn8_act = tf.nn.relu(bn8)
	hidden9 = tf.layers.dense(bn8_act,25,name='hidden9')
	bn9 = my_batch_norm_layer(hidden9)
	bn9_act = tf.nn.relu(bn9)
	hidden10 = tf.layers.dense(bn9_act,10,name='hidden10')
	bn10 = my_batch_norm_layer(hidden10)
	bn10_act = tf.nn.relu(bn10)
	hidden11 = tf.layers.dense(bn10_act,5,name='hidden11')
	bn11 = my_batch_norm_layer(hidden11)
	bn11_act = tf.nn.relu(bn11)
	hidden12 = tf.layers.dense(bn11_act,100,name='hidden12')
	bn12 = my_batch_norm_layer(hidden11)
	bn12_act = tf.nn.relu(bn12)
	logits_before_batch = tf.layers.dense(bn10_act,n_outputs,name='outputs')
	logits = my_batch_norm_layer(logits_before_batch)

with tf.name_scope('loss'):
	xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
															logits=logits)
	loss = tf.reduce_mean(xentropy,name='loss')

with tf.name_scope('optimizer'):
	optimizer = tf.train.AdamOptimizer()
	training_op = optimizer.minimize(loss)

with tf.name_scope('evaluate'):
	correct = tf.nn.in_top_k(logits,y,1)
	accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
minimal = 0
i = 0
'''
with tf.Session() as sess:
	init.run()
	for epoch in range(n_epochs):
		i += 1
		sess.run(training_op,feed_dict={X:X_ready,y:y_ready})
		if epoch%10 == 0 :
			acc_val = accuracy.eval(feed_dict={X:X_ready,y:y_ready})
			print('Accuracy on the test set is - {}'.format(acc_val))
			if acc_val > minimal:
				minimal = acc_val
				b = i
				save_path = saver.save(sess,'./my_model_final.ckpt')
	print(minimal,'This is my NNet!')
	print('This is number of epoch - ',b)
'''

X_new = pd.read_csv('test.csv')
# Подготавливаем к алгоритму
numericals2 = prepare_dataset(columns,X_new)
X_new_ready = merge_vectors(numericals2)
X_new_ready = X_new_ready.drop('PassengerId',axis=1)

#Нумерация Id
data1 = pd.read_csv('gender_submission.csv')
Id = data1.drop('Survived',axis=1)

with tf.Session() as sess:
	saver.restore(sess,'./my_model_final.ckpt')
	Z = logits.eval(feed_dict={X:X_new_ready})
	y_pred = np.argmax(Z,axis=1)

y_pred = pd.DataFrame(y_pred,columns=['Survived'])
y_pred = pd.merge(Id,y_pred,left_index=True,
										right_index=True,how='left')
w = y_pred.to_csv('answer.csv',index=False)



