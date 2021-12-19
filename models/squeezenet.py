import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Convolution2D, Conv2D, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, Activation, ReLU, concatenate, Concatenate, UpSampling2D, \
    Dropout, BatchNormalization, Lambda
from tensorflow.keras.models import Model

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"


# Modular function for Fire Node.
def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


def squeezenet(image_shape=(256, 256, 3),
               use_bn_on_input=False,
               first_stride=2):
    raw_image_input = tf.keras.Input(shape=image_shape)
    if use_bn_on_input:
        image_input = BatchNormalization()(raw_image_input)
    else:
        image_input = raw_image_input

    x = Convolution2D(64, (3, 3), strides=(first_stride, first_stride), padding='same', name='conv1')(image_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)

    x = Convolution2D(1000, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)

    model = Model(image_input, out, name='squeezenet')
    return model


if __name__ == '__main__':
    snet = squeezenet(image_shape=(8, 8, 12))
    snet.summary()
