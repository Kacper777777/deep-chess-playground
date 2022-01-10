import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Convolution2D, Conv2D, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, Activation, ReLU, concatenate, Concatenate, UpSampling2D, \
    Dropout, BatchNormalization, Lambda, Flatten, Dense
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


def squeezenet_feature_extractor(image_shape=(8, 8, 18),
                                 use_bn_on_input=False,
                                 first_stride=1,
                                 name='squeezenet'):
    raw_image_input = tf.keras.Input(shape=image_shape)
    if use_bn_on_input:
        image_input = BatchNormalization()(raw_image_input)
    else:
        image_input = raw_image_input

    x = Convolution2D(64, (3, 3), strides=(first_stride, first_stride), padding='same', name='conv1')(image_input)
    x = Activation('relu', name='relu_conv1')(x)
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)

    model = Model(image_input, x, name=name)
    return model


def squeezenet_chess_move_classifier(image_shape=(8, 8, 18)):
    chessboard_before = tf.keras.Input(shape=image_shape, name='chessboard_before')
    chessboard_after = tf.keras.Input(shape=image_shape, name='chessboard_after')
    x = Concatenate(axis=-1, name='concatenated_positions')([chessboard_before, chessboard_after])
    feature_extractor = squeezenet_feature_extractor(image_shape=(image_shape[0], image_shape[1], image_shape[2]*2))
    features = feature_extractor(x)
    x = Convolution2D(1, (1, 1), padding='valid', activation='relu', name='last_conv')(features)
    x = GlobalAveragePooling2D()(x)
    out = Activation('sigmoid', name='target')(x)
    model = Model(inputs=[chessboard_before, chessboard_after], outputs=out)
    return model


if __name__ == '__main__':
    schess = squeezenet_chess_move_classifier(image_shape=(8, 8, 18))
    schess.summary()
