import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, Activation, Multiply, Concatenate

def build_keras_unet(input_shape=(512, 512, 4)):
    """
    Exact topological recreation of the user's Attention U-Net 
    to successfully map all 24 layers of weights from the corrupted .h5.
    Explicitly named to allow `load_weights(..., by_name=True)`.
    """
    inputs = Input(input_shape, name='input_layer_1')

    # --- Encoder Block 1 ---
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_24')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_25')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_3')(c1)

    # --- Encoder Block 2 ---
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_26')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_27')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_4')(c2)

    # --- Encoder Block 3 ---
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_28')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_29')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2), name='max_pooling2d_5')(c3)

    # --- Bottleneck ---
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_30')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_31')(c4)

    # ==========================
    # --- Decoder Block 1 ---
    # ==========================
    u1 = UpSampling2D(size=(2, 2), name='up_sampling2d_3')(c4)
    
    # Attention Gate 1
    # conv2d_32 expects (1, 1, 256, 256) -> input is c3 (256)
    # conv2d_33 expects (1, 1, 512, 256) -> input is u1 (512)
    x1 = Conv2D(256, (1, 1), activation='linear', padding='valid', name='conv2d_32')(c3)
    g1 = Conv2D(256, (1, 1), activation='linear', padding='valid', name='conv2d_33')(u1)
    add1 = Add(name='add_3')([g1, x1])
    act1 = Activation('relu', name='activation_3')(add1)
    psi1 = Conv2D(1, (1, 1), activation='sigmoid', padding='valid', name='conv2d_34')(act1)
    mult1 = Multiply(name='multiply_3')([psi1, c3])     
    concat1 = Concatenate(name='concatenate_3')([u1, mult1])
    
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_35')(concat1)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_36')(c5)

    # ==========================
    # --- Decoder Block 2 ---
    # ==========================
    u2 = UpSampling2D(size=(2, 2), name='up_sampling2d_4')(c5)
    
    # Attention Gate 2
    # conv2d_37 expects (1, 1, 128, 128) -> input is c2 (128)
    # conv2d_38 expects (1, 1, 256, 128) -> input is u2 (256)
    x2 = Conv2D(128, (1, 1), activation='linear', padding='valid', name='conv2d_37')(c2)
    g2 = Conv2D(128, (1, 1), activation='linear', padding='valid', name='conv2d_38')(u2)
    add2 = Add(name='add_4')([g2, x2])
    act2 = Activation('relu', name='activation_4')(add2)
    psi2 = Conv2D(1, (1, 1), activation='sigmoid', padding='valid', name='conv2d_39')(act2)
    mult2 = Multiply(name='multiply_4')([psi2, c2])     
    concat2 = Concatenate(name='concatenate_4')([u2, mult2])
    
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_40')(concat2)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_41')(c6)

    # ==========================
    # --- Decoder Block 3 ---
    # ==========================
    u3 = UpSampling2D(size=(2, 2), name='up_sampling2d_5')(c6)
    
    # Attention Gate 3
    # conv2d_42 expects (1, 1, 64, 64) -> input is c1 (64)
    # conv2d_43 expects (1, 1, 128, 64) -> input is u3 (128)
    x3 = Conv2D(64, (1, 1), activation='linear', padding='valid', name='conv2d_42')(c1)
    g3 = Conv2D(64, (1, 1), activation='linear', padding='valid', name='conv2d_43')(u3)
    add3 = Add(name='add_5')([g3, x3])
    act3 = Activation('relu', name='activation_5')(add3)
    psi3 = Conv2D(1, (1, 1), activation='sigmoid', padding='valid', name='conv2d_44')(act3)
    mult3 = Multiply(name='multiply_5')([psi3, c1])     
    concat3 = Concatenate(name='concatenate_5')([u3, mult3])
    
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_45')(concat3)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2d_46')(c7)

    # ==========================
    # --- Output ---
    # ==========================
    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='valid', name='conv2d_47')(c7)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model
