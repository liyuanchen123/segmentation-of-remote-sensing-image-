import keras
from keras import Model
from keras.layers import (Activation, Add, AveragePooling2D,
                          BatchNormalization, Conv2D, Conv2DTranspose, Flatten,
                          Input, MaxPool2D, Reshape, UpSampling2D,
                          ZeroPadding2D, concatenate,Dropout)
import keras.backend as backend
import keras.utils as keras_utils

'''
def get_segmentation_model(input, output):

    img_input = input
    o = output

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    if IMAGE_ORDERING == 'channels_first':
        output_height = o_shape[2]
        output_width = o_shape[3]
        input_height = i_shape[2]
        input_width = i_shape[3]
        n_classes = o_shape[1]
        o = (Reshape((-1, output_height*output_width)))(o)
        o = (Permute((2, 1)))(o)
    elif IMAGE_ORDERING == 'channels_last':
        output_height = o_shape[1]
        output_width = o_shape[2]
        input_height = i_shape[1]
        input_width = i_shape[2]
        n_classes = o_shape[3]
        o = (Reshape((output_height*output_width, -1)))(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.model_name = ""

    model.train = MethodType(train, model)
    model.predict_segmentation = MethodType(predict, model)
    model.predict_multiple = MethodType(predict_multiple, model)
    model.evaluate_segmentation = MethodType(evaluate, model)

    return model
'''
# Decoder for UNet is adapted from keras-segmentation
# https://github.com/divamgupta/image-segmentation-keras/blob/master/keras_segmentation/models/unet.py
def UNet( f4, f3, f2, f1, output_height, output_width, l1_skip_conn=True, n_classes=6):
    o = f4

    IMAGE_ORDERING = 'channels_last'
    if IMAGE_ORDERING == 'channels_first':
        MERGE_AXIS = 1
    elif IMAGE_ORDERING == 'channels_last':
        MERGE_AXIS = -1
        
#    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
#    o = (concatenate([o, f4], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
#    o = Dropout(0.3)(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)
#    o = Dropout(0.3)(o)
#    o = (UpSampling2D((2,2), data_format=IMAGE_ORDERING))(o)
#    o = (Conv2D(32, (3, 3), padding='same', activation='relu', data_format=IMAGE_ORDERING))(o)
#    o = (UpSampling2D((2,2), data_format=IMAGE_ORDERING))(o)
#    o = (Conv2D(32, (3, 3), padding='same', activation='relu', data_format=IMAGE_ORDERING))(o)
    
    o = (UpSampling2D((4,4), data_format=IMAGE_ORDERING))(o)
    o = Conv2D(n_classes, 1, activation="softmax")(o)
    

#    o = Conv2D(n_classes, (3, 3), padding='same',
#               data_format=IMAGE_ORDERING)(o)

    

    # o = (Reshape((output_height*output_width, -1)))(o)

#    o = (Activation('softmax'))(o)

    return o

def UNet1( f5,f4, f3, f2,f1 , output_height, output_width, n_classes=6):
    channels = [64, 128, 256, 512]

    # 32, 32, 512 -> 64, 64, 512
    P5_up = UpSampling2D(size=(2, 2))(f5)
    # 64, 64, 512 + 64, 64, 512 -> 64, 64, 1024
    P4 = concatenate([f4, P5_up],axis=-1)
    # 64, 64, 1024 -> 64, 64, 512
    P4 = Conv2D(channels[3], 3, padding='same', kernel_initializer='he_normal')(f4)
    
#    P4 = Conv2D(channels[3], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P4)
    P4 = (BatchNormalization())(P4)
    P4 = Activation('relu')(P4)
    
    # 64, 64, 512 -> 128, 128, 512
    P4_up = UpSampling2D(size=(2, 2))(P4)
    # 128, 128, 256 + 128, 128, 512 -> 128, 128, 768
    P3 = concatenate([f3, P4_up],axis=-1)
    # 128, 128, 768 -> 128, 128, 256
    P3 = Conv2D(channels[2], 3,  padding='same', kernel_initializer='he_normal')(P3)
#    P3 = Conv2D(channels[2], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P3)
    P3 = (BatchNormalization())(P3)
    P3 = Activation('relu')(P3)
    
    # 128, 128, 256 -> 256, 256, 256
    P3_up = UpSampling2D(size=(2, 2))(P3)
    # 256, 256, 256 + 256, 256, 128 -> 256, 256, 384
    P2 = concatenate([f2, P3_up],axis=-1)
    # 256, 256, 384 -> 256, 256, 128
    P2 = Conv2D(channels[1], 3,  padding='same', kernel_initializer='he_normal')(P2)
#    P2 = Conv2D(channels[1], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P2)
    P2 = (BatchNormalization())(P2)
    P2 = Activation('relu')(P2)
    
    # 256, 256, 128 -> 512, 512, 128
    P2_up = UpSampling2D(size=(2, 2))(P2)
#    # 512, 512, 128 + 512, 512, 64 -> 512, 512, 192
    P1 = concatenate([f1, P2_up],axis=-1)
#    # 512, 512, 192 -> 512, 512, 64
    P1 = Conv2D(channels[0], 3, padding='same', kernel_initializer='he_normal')(P1)
#    P1 = Conv2D(channels[0], 3, activation='relu', padding='same', kernel_initializer='he_normal')(P1)
    P1 = (BatchNormalization())(P2_up)
    P2 = Activation('relu')(P2)
    
    # 512, 512, 64 -> 512, 512, num_classes
    P1 = Conv2D(n_classes, 1, activation="softmax")(P1)
    
    return P1