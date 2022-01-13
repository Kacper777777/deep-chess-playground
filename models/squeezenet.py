import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Convolution2D, Conv2D, MaxPooling2D, \
    GlobalAveragePooling2D, Activation, ReLU, concatenate, Concatenate, \
    Dropout, BatchNormalization, Lambda, Flatten, Dense
from tensorflow.keras.models import Model
from models.heads import move_classification_head, position_evaluation_head

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


def chessboard_feature_extractor(image_shape=(8, 8, 18),
                                 name='squeezenet'):
    chessboard_input = tf.keras.Input(shape=image_shape)

    x = Convolution2D(64, (3, 3), strides=(1, 1), padding='same', name='conv1')(chessboard_input)
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

    model = Model(chessboard_input, x, name=name)
    return model


def chess_move_classifier(input_shape=(8, 8, 18)):
    chessboard = tf.keras.Input(shape=input_shape, name='chessboard')
    feature_extractor = chessboard_feature_extractor(image_shape=input_shape)
    features = feature_extractor(chessboard)
    out = move_classification_head(features)
    model = Model(inputs=chessboard, outputs=out)
    return model


def chess_position_evaluator(input_shape=(8, 8, 18)):
    chessboard = tf.keras.Input(shape=input_shape, name='chessboard')
    feature_extractor = chessboard_feature_extractor(image_shape=input_shape)
    features = feature_extractor(chessboard)
    out = position_evaluation_head(features)
    model = Model(inputs=chessboard, outputs=out)
    return model


def chess_move_classifier_and_position_evaluator(input_shape=(8, 8, 18)):
    chessboard = tf.keras.Input(shape=input_shape, name='chessboard')
    feature_extractor = chessboard_feature_extractor(image_shape=input_shape)
    features = feature_extractor(chessboard)
    policy = move_classification_head(features)
    value = position_evaluation_head(features)
    model = Model(inputs=chessboard, outputs=[policy, value])
    return model


if __name__ == '__main__':
    neural_move_classifier = chess_move_classifier(input_shape=(8, 8, 18))
    neural_position_evaluator = chess_position_evaluator(input_shape=(8, 8, 18))
    neural_move_classifier_and_position_evaluator = chess_move_classifier_and_position_evaluator(input_shape=(8, 8, 18))
    neural_move_classifier.summary()
    neural_position_evaluator.summary()
    neural_move_classifier_and_position_evaluator.summary()
