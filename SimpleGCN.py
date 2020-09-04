import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
#from scipy.sparse import csr_matrix
from collections import namedtuple # tensorflow2.x 在layer计算时会将float64(double)转为float32(float)
#tf.keras.backend.set_floatx('float64')
DataSet = namedtuple(
    'DataSet',
    field_names=['A','x_train', 'y_train', 'x_test', 'y_test']
)
def load_karate_club():
    edgelist = np.loadtxt("data/karate.edgelist", dtype=int)
    features = np.loadtxt("data/karate.attributes.csv", dtype=str, skiprows=1, delimiter=',')
    features = features[:, 1:3]
    x_train, y_train = map(np.array, zip(
        *[([index], features[index][1] == 'Administrator') for index, role in enumerate(features[:, 0]) if
          role in {'Administrator', 'Instructor'}]))
    x_test, y_test = map(np.array, zip(
        *[([index], features[index][1] == 'Administrator') for index, role in enumerate(features[:, 0]) if
          role == 'Member']))
    x_test = np.array(x_test).flatten()
    x_train = np.array(x_train).flatten()
    # row = edgelist[:, 0]
    # col = edgelist[:, 1]
    # data = np.ones(row.shape[0], dtype=int)
    # A = sp.csr_matrix((data, (row, col)), shape=(features.shape[0], features.shape[0])).toarray()
    A=np.zeros((features.shape[0],features.shape[0]),dtype='float32')
    for row in edgelist:
        A[row[0]][row[1]]=1.
    return DataSet(A,x_train,y_train,x_test,y_test)

def pre_processing_adj(A):
    # A_eye=A+sp.eye(A.shape[0])
    # rowsum = np.array(A_eye.sum(1))
    # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    A_eye = A + np.eye(A.shape[0])
    rowsum = np.array(A_eye.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return A_eye.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype('float32')

class SpectralLayer(tf.keras.layers.Layer):
    def __init__(self,A,units,name=None,activation=None):
        super().__init__(name=name)
        self.A=A
        self.units=units
        self.activation=activation


    def build(self,input_shape):
        initializer=tf.keras.initializers.GlorotNormal()
        #self.w = tf.Variable(tf.ones([input_shape[-1], self.units]),name='w')
        self.w=tf.Variable(initializer(shape=(input_shape[-1], self.units)),name='w')

    def call(self, input):
        AX=tf.matmul(self.A,input)
        if self.activation=='tanh':
            output=tf.nn.tanh(tf.matmul(AX,self.w))
        if self.activation == 'relu':
            output = tf.nn.relu(tf.matmul(AX, self.w))
        return output

class LogicRegressor(tf.keras.layers.Layer):
    def __init__(self,name=None):
        super().__init__(name=name)


    def build(self,input_shape):
        #self.w = np.zeros((input_shape[1], 1))
        #self.w = np.ones((input_shape[1], 1))
        initializer = tf.keras.initializers.GlorotNormal()
        self.w = tf.Variable(initializer(shape=(input_shape[-1], 1)), name='w')
        self.b=tf.Variable(tf.zeros([1]),name='b')

    def call(self,input):
        output=tf.matmul(input,self.w)+self.b
        return tf.nn.sigmoid(output)

class GCNModel(tf.keras.models.Model):
    def __init__(self,A,name=None):
        super().__init__(name=name)
        self.A=A
        self.spectralLayer1=SpectralLayer(self.A,4,name='spectralLayer1',activation='tanh')
        self.spectralLayer2=SpectralLayer(self.A,2,name='spectralLayer2',activation='tanh')
        self.classifyLayer=LogicRegressor(name='classifyLayer')

    def call(self,input):
        result_s1=self.spectralLayer1(input)
        result_s2=self.spectralLayer2(result_s1)
        result_c=self.classifyLayer(result_s2)
        #return result_s1,result_s2,result_c
        return result_c

def cross_loss(target_y, predicted_y):
  return tf.reduce_mean(tf.square(target_y - predicted_y))

def train(model, X, x_train, y_train, learning_rate, epochs):
    for epoch in range(1, epochs + 1):
        cur_loss = 0.
        cur_preds = []
        for i, x in enumerate(x_train):
            # Trainable variables are automatically tracked by GradientTape
            # Use GradientTape to calculate the gradients with respect to W and b
            with tf.GradientTape() as t:
                preds = model(X)[x]
                loss = cross_loss(y_train[i], preds)

            dw1, dw2, dcw, dcb = t.gradient(loss,
                                            [model.spectralLayer1.w, model.spectralLayer2.w, model.classifyLayer.w,
                                             model.classifyLayer.b])
            # Subtract the gradient scaled by the learning rate
            model.spectralLayer1.w.assign_sub(learning_rate * dw1)
            model.spectralLayer2.w.assign_sub(learning_rate * dw2)
            model.classifyLayer.w.assign_sub(learning_rate * dcw)
            model.classifyLayer.b.assign_sub(learning_rate * dcb)

            cur_loss += loss
            cur_preds += [preds]
            #cur_preds.append(np.array(preds))
        if (epoch % (epochs // 10)) == 0:
            print(f"Epoch {epoch}/{epochs} -- Loss: {cur_loss: .4f}")
            print("cum_preds:", cur_preds)

def predict(model, X, nodes):
    preds=np.array(model(X)).flatten()[nodes]
    return np.where(preds >= 0.5, 1, 0)

def accuracy(y_true,y_predcit):
    num=0
    for i in range(len(y_true)):
        if y_true[i]==y_predcit[i]:
            num+=1
    return num/len(y_true)
# initial
zkc=load_karate_club()
A=zkc.A
A=pre_processing_adj(A)#.todense()
x_train=zkc.x_train.tolist()
y_train=zkc.y_train.tolist()
x_test=zkc.x_test.tolist()
y_test=zkc.y_test.tolist()
X=tf.eye(A.shape[0])
model=GCNModel(A,name="myModel")
# train use custom method
#train(model,X,x_train,y_train,0.1,100)
test_result=predict(model,X,x_test)
print(accuracy(y_test,test_result))
#train use keras's API
# model.compile(
#     run_eagerly=False,
#     optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
#     loss=tf.keras.losses.mean_squared_error,
# )
# model.fit(x_test, y_test, epochs=100, batch_size=1)
#
# print model
print(model.variables)
# print(model.submodules)
#print(model.summary())
