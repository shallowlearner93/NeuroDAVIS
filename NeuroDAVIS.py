import numpy as np
import tensorflow as tf
import keras
import keras.optimizers
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers


# neuroDAVIS architecture
def NeuroDAVIS(data, level, dim, lambda_act, lambda_weight, num_neuron, bs, epoch, sd, verbose):
    num_in_neuron = data.shape[0]
    num_out_neuron = level.shape[1]
    
    inputs = tf.keras.Input(shape=(num_in_neuron,))

    layer1 = keras.layers.Dense(dim, activation="linear",
                      activity_regularizer=regularizers.l2(lambda_act),
                      kernel_regularizer=regularizers.l2(lambda_weight), 
                      kernel_initializer=tf.keras.initializers.RandomNormal(seed=sd),
                      )(inputs)
    
    layer2 = keras.layers.Dense(num_neuron[0], activation="relu",
                      activity_regularizer=regularizers.l2(lambda_act),
                      kernel_regularizer=regularizers.l2(lambda_weight),
                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=sd),#he_uniform(seed=sd),
                      bias_initializer = tf.keras.initializers.Constant(0.01)
                      )(layer1)
       
    layer3 = keras.layers.Dense(num_neuron[1], activation="relu",
                      activity_regularizer=regularizers.l2(lambda_act),
                      kernel_regularizer=regularizers.l2(lambda_weight),
                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=sd),#he_uniform(seed=sd),
                      bias_initializer = tf.keras.initializers.Constant(0.01)
                      )(layer2)

    outputs = keras.layers.Dense(num_out_neuron, activation="linear",
                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=sd),
                      bias_initializer = tf.keras.initializers.Zeros()
                      )(layer3)

    neuroDAVIS = keras.Model(inputs=inputs, outputs=outputs)
    low = keras.Model(inputs=inputs, outputs=layer1)

    neuroDAVIS.compile(
        loss=keras.losses.mean_squared_error,
        optimizer=tf.keras.optimizers.Adam()
    )
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30)
    neuroDAVIS.fit(data, level, batch_size=bs, epochs=epoch, verbose=verbose, callbacks=[callback])       

    return neuroDAVIS, low