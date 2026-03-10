"""
Keras Attention U-Net
======================
Exact replica of the Colab training architecture so that positional
weight loading (load_weights without by_name) works correctly.
"""

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, Activation, Multiply, Concatenate

def build_keras_unet(input_shape=(512, 512, 4)):
    # Explicitly named to match the .h5 model summary perfectly
    inputs = Input(input_shape, name='input_layer')

    # --- Encoder Block 1 ---
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_1')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d')(c1)

    # --- Encoder Block 2 ---
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_2')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_3')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_1')(c2)

    # --- Encoder Block 3 ---
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_4')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_5')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_2')(c3)

    # --- Bottleneck ---
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_6')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_7')(c4)

    # ==========================
    # --- Decoder Block 1 ---
    # ==========================
    u1 = UpSampling2D(size=(2, 2), name='up_sampling2d')(c4)
    
    # Attention Gate 1
    x1 = Conv2D(256, (1, 1), activation='linear', padding='valid', name='conv2d_8')(c3)
    g1 = Conv2D(256, (1, 1), activation='linear', padding='valid', name='conv2d_9')(u1)
    
    add1 = Add(name='add')([x1, g1]) 
    act1 = Activation('relu', name='activation')(add1)
    psi1 = Conv2D(1, (1, 1), activation='sigmoid', padding='valid', name='conv2d_10')(act1)
    
    mult1 = Multiply(name='multiply')([c3, psi1])     
    concat1 = Concatenate(name='concatenate')([u1, mult1]) # Correctly creates the 768 channel map
    
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_11')(concat1)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_12')(c5)

    # ==========================
    # --- Decoder Block 2 ---
    # ==========================
    u2 = UpSampling2D(size=(2, 2), name='up_sampling2d_1')(c5)
    
    # Attention Gate 2
    x2 = Conv2D(128, (1, 1), activation='linear', padding='valid', name='conv2d_13')(c2)
    g2 = Conv2D(128, (1, 1), activation='linear', padding='valid', name='conv2d_14')(u2)
    
    add2 = Add(name='add_1')([x2, g2])
    act2 = Activation('relu', name='activation_1')(add2)
    psi2 = Conv2D(1, (1, 1), activation='sigmoid', padding='valid', name='conv2d_15')(act2)
    
    mult2 = Multiply(name='multiply_1')([c2, psi2])     
    concat2 = Concatenate(name='concatenate_1')([u2, mult2])
    
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_16')(concat2)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_17')(c6)

    # ==========================
    # --- Decoder Block 3 ---
    # ==========================
    u3 = UpSampling2D(size=(2, 2), name='up_sampling2d_2')(c6)
    
    # Attention Gate 3
    x3 = Conv2D(64, (1, 1), activation='linear', padding='valid', name='conv2d_18')(c1)
    g3 = Conv2D(64, (1, 1), activation='linear', padding='valid', name='conv2d_19')(u3)
    
    add3 = Add(name='add_2')([x3, g3])
    act3 = Activation('relu', name='activation_2')(add3)
    psi3 = Conv2D(1, (1, 1), activation='sigmoid', padding='valid', name='conv2d_20')(act3)
    
    mult3 = Multiply(name='multiply_2')([c1, psi3])     
    concat3 = Concatenate(name='concatenate_2')([u3, mult3])
    
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_21')(concat3)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_22')(c7)

    # ==========================
    # --- Output ---
    # ==========================
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='valid', name='conv2d_23')(c7)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model
