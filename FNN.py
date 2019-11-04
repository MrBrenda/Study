#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xlrd
import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Activation
from keras import backend as K
from sklearn.model_selection import train_test_split

def Load_train_XY(k,n,m,dianwei):
    filename = 'D:/PyProjects/jupyter_notebook/WQ_time_series_Pre/MinMax_training_sets/'+str(k)+'/'+str(n)+'-'+str(m)+'/'+dianwei+'-'+str(n)+'-'+str(m)+'.csv'
    dataset = pd.read_csv(filename, header=None, engine='python')
    Y = np.array(dataset)[:, list(range(4))]
    X = np.array(dataset)[:, list(range(4, np.shape(dataset)[1]))]
    train_X, validate_X, train_Y, validate_Y = train_test_split(X, Y, test_size=0.4, random_state=0)
    return train_X, validate_X, train_Y, validate_Y

def Build_NN_model(input_dim, output_dim, hidden_layers, hidden_dim,
                   activation, learning_rate):
    NN_model = Sequential()
    NN_model.add(Dense(units=int(hidden_dim), input_dim=input_dim, use_bias= True))
    NN_model.add(Activation(activation))
    for i in range(hidden_layers-1):
        NN_model.add(Dense(units=int(hidden_dim), use_bias= True))
        NN_model.add(Activation(activation))
    # 输出层
    NN_model.add(Dense(units=int(output_dim)))
    NN_model.add(Activation(activation))
    optimizer_here = optimizers.Adam(lr=learning_rate, beta_2=0.99)
    NN_model.compile(loss='mse', optimizer=optimizer_here)
    return NN_model

def train_models(k, n, m, dianwei):
    # load 数据
    train_X, validate_X, train_Y, validate_Y = Load_train_XY(k, n, m, dianwei)

    # 定义超参数
    input_dim = train_X.shape[1]
    output_dim = train_Y.shape[1]
    hidden_layers_list = list(range(1, 9))  # 8
    hidden_dim_list = list(range(4, 40, 4))  # 9
    activation_list = ['tanh', 'relu']  # 2
    learning_rate_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]  # 6
    batch_size_list = [2, 4, 8, 16, 32, 64]  # 6
    # 相乘得：8*9*2*6*6 = 5184

    # 循环训练模型
    for hidden_layers in hidden_layers_list:
        for hidden_dim in hidden_dim_list:
            for activation in activation_list:
                for learning_rate in learning_rate_list:
                    for batch_size in batch_size_list:
                        NN_model = Build_NN_model(input_dim, output_dim, hidden_layers, hidden_dim,
                                                  activation, learning_rate)
                        history = NN_model.fit(train_X, train_Y, epochs=100, batch_size=batch_size, verbose=0)
                        validate_mse = NN_model.evaluate(validate_X, validate_Y)
                        s = str(hidden_layers)+','+str(hidden_dim)+','+activation+','+str(learning_rate)+\
                            ','+str(batch_size)+','+str(history.history['loss'][-1])+','+str(validate_mse)+'\n'
                        print(s)
                        with open('D:\PyProjects\jupyter_notebook\WQ_time_series_Pre\Results\MSEs/'+\
                                  dianwei+'-'+str(k)+'-'+str(n)+'-'+str(m)+'.txt', 'a') as f:
                            f.write(s)
                        K.clear_session()
                        tf.reset_default_graph()
for k in [100,250,500]:
    for n in [1,2,3,4]:
        train_models(k,n,1,'安徽巢湖裕溪口')