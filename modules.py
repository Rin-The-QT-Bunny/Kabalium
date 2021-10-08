"""
Title: Data-Loader Module
"""

import tensorflow as tf # basic utils of tensorflow
import tensorflow.keras as keras # keras API installed
import tensorflow.keras.layers as layers # Default Layers

import scipy # serious data related modules
import numpy as np # Linear Algebra
import matplotlib.pyplot as plt # Import matploylib to plot graphs etc.
from scipy import stats

class GMM(object):
    def __init__(self, k: int, d: int):
        '''
        k: K值
        d: 样本属性的数量
        '''
        self.K = k
        # 初始化参数
        self.p = np.random.rand(k)
        self.p = self.p / self.p.sum()      # 保证所有p_k的和为1
        self.means = np.random.rand(k, d)
        self.covs = np.empty((k, d, d))
        for i in range(k):                  # 随机生成协方差矩阵，必须是半正定矩阵
            self.covs[i] = np.eye(d) * np.random.rand(1) * k

    def fit(self, data: np.ndarray):
        '''
        data: 数据矩阵，每一行是一个样本，shape = (N, d)
        '''
        for _ in range(100):
            density = np.empty((len(data), self.K))
            for i in range(self.K):
                # 生成K个概率密度函数并计算对于所有样本的概率密度
                norm = stats.multivariate_normal(self.means[i], self.covs[i])
                density[:,i] = norm.pdf(data)
            # 计算所有样本属于每一类别的后验
            posterior = density * self.p
            posterior = posterior / posterior.sum(axis=1, keepdims=True)
            # 计算下一时刻的参数值
            p_hat = posterior.sum(axis=0)
            mean_hat = np.tensordot(posterior, data, axes=[0, 0])
            # 计算协方差
            cov_hat = np.empty(self.covs.shape)
            for i in range(self.K):
                tmp = data - self.means[i]
                cov_hat[i] = np.dot(tmp.T*posterior[:,i], tmp) / p_hat[i]
            # 更新参数
            self.covs = cov_hat
            self.means = mean_hat / p_hat.reshape(-1,1)
            self.p = p_hat / len(data)

        print(self.p)
        print(self.means)
        print(self.covs)

class General_Graph:
    # Construct the general graph
    def __init__(self,backgrounds,nodes,relations):
        self.P = backgrounds #Global variable of the graph
        self.X = nodes # Nodes representation
        self.R = relations # Vertex Information\
        print("Graph Representation Module Done")

        # Set up the dimension transformer
        self.dim = backgrounds.shape[0]
        print("Representation Dimension is {}".format(self.dim))

        self.NNP = self.construct_evolution_P(self.dim*2,self.dim)
        self.NNQ = self.construct_evolution_Q(self.dim*4,self.dim)
        self.NNR = self.construct_evolution_R(self.dim*3,self.dim)
        print("Neural Constructors are Created")
    
    def construct_evolution_P(self,in_dim,dim):
        # Input is the representation of nodes and concepts
        x_in_p = keras.Input(shape = (in_dim))

        # construct hidden units and layers
        hid1_p = layers.Dense(32,"tanh")(x_in_p)
        hid2_p = layers.Dense(64,"relu")(hid1_p)
        hid3_p = layers.Dense(32,"tanh")(hid2_p)

        # reconstruction loss between x_io and x_out
        x_out_p = layers.Dense(dim,"sigmoid")(hid3_p)

        # Build the hidden Model
        return keras.Model(x_in_p,x_out_p)

    def construct_evolution_Q(self,in_dim,dim):
        # Input is the representation of nodes and concepts
        x_in_q = keras.Input(shape = (in_dim))

        # construct hidden units and layers
        hid1_q = layers.Dense(32,"tanh")(x_in_q)
        hid2_q = layers.Dense(64,"relu")(hid1_q)
        hid3_q = layers.Dense(32,"tanh")(hid2_q)

        # reconstruction loss between x_io and x_out
        x_out_q = layers.Dense(dim,"sigmoid")(hid3_q)

        # Build the hidden Model
        return keras.Model(x_in_q,x_out_q)
    
    def construct_evolution_R(self,in_dim,dim):
        # Input is the representation of nodes and concepts
        x_in_q = keras.Input(shape = (in_dim))

        # construct hidden units and layers
        hid1_q = layers.Dense(32,"tanh")(x_in_q)
        hid2_q = layers.Dense(64,"relu")(hid1_q)
        hid3_q = layers.Dense(32,"tanh")(hid2_q)

        # reconstruction loss between x_io and x_out
        x_out_q = layers.Dense(dim,"sigmoid")(hid3_q)

        # Build the hidden Model
        return keras.Model(x_in_q,x_out_q)


    def evolve_global(self,X):
        # Evolve the background information and representation
        tensor_background = self.P
        reduce_nodes = tf.reduce_max(X,axis = 0)
        conct = tf.concat([tensor_background,reduce_nodes],axis = 0)
        # Concatenate the information and combine them
        conct = tf.reshape(conct,[1,conct.shape[0]])
        P_next = self.NNP(conct)

        return P_next

    def evolve_nodes(self,X):

        # Nodes to evolve
        T_nodes  = []
        # Evolve the nodes information and representation
        tensor_background = self.P
        
        for i in range(X.shape[0]):
            reduct_R1 = tf.reduce_max(self.R[i,:,:],0)
            reduct_R2 = tf.reduce_max(self.R[:,i,:],0)
            reduct_R1 = tf.reshape(reduct_R1,[16])
            reduct_R2 = tf.reshape(reduct_R2,[16])
            tensor_background = tf.reshape(tensor_background,[16])
            signal = tf.reshape(X[i],[16])

            raw = tf.concat([tensor_background,signal,reduct_R1,reduct_R2],0)
            raw = tf.reshape(raw,[1,64])
            echo = self.NNQ(raw)
            echo = tf.reshape(echo,[16])
            T_nodes.append(echo)
        
        # Convert the representation to nodes
        T_nodes = tf.convert_to_tensor(T_nodes)
        return T_nodes
    
    def evolve_relations(self,X,R):
        N = self.X.shape[0]
        return_relations = np.zeros([N,N,X.shape[1]])

        for i in range(N):
            for j in range(N):



                conct = tf.concat([R[i,j,:],X[i],X[j]],axis = 0)
                conct = tf.reshape(conct,[1,-1])
                representation = self.NNR(conct)
                return_relations[i][j] = representation
        return_relations = tf.convert_to_tensor(return_relations)

        return return_relations