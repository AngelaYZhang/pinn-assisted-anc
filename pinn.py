#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PINN model training setup for sound field interpolation

@author: Yile Angela Zhang
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pickle
from scipy.io import wavfile
import scipy.optimize
from datetime import datetime
import scipy.io as sio
import sys
tf.keras.backend.set_floatx('float64')


##########################################################################################
# PINN definition



# Set-up PINN architecture
##########################################
def interpolate_PINN(num_input=4, num_hidden_layers=3, neurons_per_layer=4):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(num_input))
    ######################################
    # # Append hidden layers
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer,
            activation=tf.keras.activations.get('tanh'),
            kernel_initializer='glorot_normal'))
    # Output 1-D being the estimated signal
    model.add(tf.keras.layers.Dense(1, dtype=tf.float64))
    return model
##########################################



# Loss terms
##########################################
def pde_loss(model, super_xyzt_input, factor_laplace,factor_x4_d2):
    ######################################
    # x1=x, x2=y, x3=z, x4=t
    with tf.GradientTape(persistent=True) as tape:
        #x1 = tf.convert_to_tensor(super_xyzt_input[:,0:1])
        #x2 = tf.convert_to_tensor(super_xyzt_input[:,1:2])
        #x3 = tf.convert_to_tensor(super_xyzt_input[:,2:3])
        #x4 = tf.convert_to_tensor(super_xyzt_input[:,3:4])
        x1, x2, x3, x4 = super_xyzt_input[:,0:1], super_xyzt_input[:,1:2], super_xyzt_input[:,2:3], super_xyzt_input[:,3:4]
        tape.watch(x1)
        tape.watch(x2)
        tape.watch(x3)
        tape.watch(x4)  
        pde_pred = model(tf.stack([x1[:,0],x2[:,0],x3[:,0],x4[:,0]],axis=1))
        x1_d1 = tape.gradient(pde_pred,x1)
        x2_d1 = tape.gradient(pde_pred,x2)
        x3_d1 = tape.gradient(pde_pred,x3)
        x4_d1 = tape.gradient(pde_pred,x4)
    ######################################
    x1_d2 = tape.gradient(x1_d1,x1)
    x2_d2 = tape.gradient(x2_d1,x2)
    x3_d2 = tape.gradient(x3_d1,x3)
    x4_d2 = tape.gradient(x4_d1,x4)
    del tape

    pde_eqn  = factor_laplace*(x1_d2+x2_d2+x3_d2) - factor_x4_d2 *x4_d2
    loss_pde = tf.reduce_mean(tf.square(pde_eqn))
    
    return loss_pde
##########################################

##########################################
def data_loss(model,sub_xyzt_input,pressure_measured):
    pressure_pred  = model(sub_xyzt_input)
    loss_data  = tf.reduce_mean(tf.square(pressure_pred-pressure_measured)) 
    return loss_data
##########################################



# Gradient calc
##########################################
def model_gradient(model,sub_xyzt_input,pressure_measured,super_xyzt_input,factor_laplace,factor_x4_d2):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(model.trainable_variables)
        loss_data = data_loss(model,sub_xyzt_input,pressure_measured)
        loss_pde = pde_loss(model,super_xyzt_input,factor_laplace,factor_x4_d2)
##########################################
        loss      = loss_data + loss_pde
##########################################
    grad = tape.gradient(loss,model.trainable_variables)
    del tape
    return loss_data, loss_pde, grad
##########################################



# Model weights update
@tf.function
##########################################
def model_fit(model,sub_xyzt_input,pressure_measured,super_xyzt_input,factor_laplace,factor_x4_d2):
    loss_data,w_loss_pde,grad =model_gradient(model,sub_xyzt_input,pressure_measured,super_xyzt_input,factor_laplace,factor_x4_d2)
    optim.apply_gradients(zip(grad,model.trainable_variables))
    return loss_data, w_loss_pde
##########################################

##########################################################################################




##########################################################################################
# Training PINN params

# Params
inputs  = 4
layers  = 1
nodes   = 16
epochs  = 500000
model   = interpolate_PINN(inputs,layers,nodes)  # input number, layer number, neurons per layer
lr      = 0.001
optim   = tf.keras.optimizers.Adam(learning_rate = lr, epsilon = 1e-9)
##########################################
# PDE parameter
c               = 343 #Speed of sound
freq            = (400+500+300)/3
wave_num        = 2*np.pi*freq/c
factor_laplace  = 1.0/(wave_num**2)

dt_true         = 1/24000 
dt_norm         = 0.3/239  #normalise t range [-0.15,0.15]
dt_adjust       = (dt_norm**2)/(dt_true**2)
factor_x4_d2    = dt_adjust / ((2*np.pi*freq)**2)

    
##########################################################################################

