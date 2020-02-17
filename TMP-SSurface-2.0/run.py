#! /usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
from keras import layers, models, optimizers
from keras.utils import to_categorical
from keras.layers import *
from keras.models import *
from keras.optimizers import SGD, Adadelta, Adam
from keras.callbacks import Callback
from keras import backend as K
K.clear_session()
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from keras import callbacks
from keras import backend as K 
from pre_processing import Processor
K.set_image_data_format('channels_last')

def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(nb_time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul

def cc(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(r)

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr    

if __name__ == "__main__":

    '''
    cmd = python run.py --fasta sample/sample.fasta --pssm_path sample/pssm/ --output_path results/
    cmd = python run.py -f sample/sample.fasta -p sample/pssm/ -o results/
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fasta', default='sample/sample.fasta')
    parser.add_argument('-p', '--pssm_path', default='sample/pssm/')
    parser.add_argument('-o', '--output_path', default='results/')
    args = parser.parse_args()
    #print(args)    
    
    processor = Processor()
    fasta = args.fasta
    pssm_path = args.pssm_path   
    output_path = args.output_path
    
    # get zpred feature
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    lr_metric = get_lr_metric(adadelta)
    
    x_zpred = processor.zpred_pre_processing(fasta, pssm_path, 25) 
    x_zpred = x_zpred.reshape(x_zpred.shape[0], 25, 41, 1)
    
    cnn_model_path = 'model/CNN_model.h5'
    cnn_model = load_model(cnn_model_path, custom_objects={'cc':cc, 'lr':lr_metric})
    cnn_model.summary()
    
    pred_zpred = cnn_model.predict(x_zpred)
 
    #parameters for LSTM
    window_length = 19
    rows, cols = window_length, 42
    nb_lstm_outputs = 700  
    nb_time_steps = window_length
    nb_input_vector = cols      
    
    # get all features
    x_test = processor.data_pre_processing(pred_zpred, fasta, pssm_path, window_length) 
    x_test = x_test.reshape(x_test.shape[0], rows, cols)
        
    # LSTM structure
    inputs = Input(shape=(nb_time_steps, nb_input_vector))
    drop1 = Dropout(0.01)(inputs)
    lstm_out = Bidirectional(LSTM(nb_lstm_outputs, return_sequences=True), name='bilstm1')(drop1)
    lstm_out2 = Bidirectional(LSTM(nb_lstm_outputs, return_sequences=True), name='bilstm2')(lstm_out)
    attention_mul = attention_3d_block(lstm_out2)
    attention_flatten = Flatten()(attention_mul)
    drop2 = Dropout(0.3)(attention_flatten)
    output = Dense(1, activation='relu')(drop2)
    model = Model(inputs=inputs, outputs=output)  
    model.summary()

    # load weights
    model.load_weights('model/LSTM_weights.h5')
    y_pred = model.predict(x_test, batch_size=128)
    
    print('finished!')
    
    # get results
    get_fasta = open(fasta)
    temp = get_fasta.readline()
    pdb_id = ""
    index = 0
    while temp:
        if(temp[0] == ">"):
            pdb_id = temp[1:].strip().split()[0]
            temp = get_fasta.readline()
            continue
        for i in temp:
            if(i != '\n'):
                rASA_line = y_pred[index][0]
                index += 1
                w = open(output_path + pdb_id + ".rASA","a+")
                w.write(str(rASA_line) + '\n')
        temp = get_fasta.readline()    

    
