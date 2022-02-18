from turtle import shape
from tensorflow import keras
from keras import layers

import config

NUMBER_OF_KERNELS_IDX = 0
KERNEL_SIZE_IDX = 1
STRIDE_IDX = 2


class IndividualColumn:

    def __init__(self, inputs, column_config: dict):
        super().__init__()
        conv_layer_1 = layers.Conv2D(filters=column_config['conv1'][NUMBER_OF_KERNELS_IDX],
                                     kernel_size=column_config['conv1'][KERNEL_SIZE_IDX],
                                     strides=column_config['conv1'][STRIDE_IDX],
                                     activation='relu',
                                     padding='same')(inputs)
        maxpool_layer_1 = layers.MaxPooling2D(pool_size=(
            column_config['maxPool1'][NUMBER_OF_KERNELS_IDX], column_config['maxPool1'][NUMBER_OF_KERNELS_IDX]),
            padding='valid')(conv_layer_1)

        conv_layer_2 = layers.Conv2D(filters=column_config['conv2'][NUMBER_OF_KERNELS_IDX],
                                     kernel_size=column_config['conv2'][KERNEL_SIZE_IDX],
                                     strides=column_config['conv2'][STRIDE_IDX],
                                     activation='relu',
                                     padding='same')(maxpool_layer_1)
        maxpool_layer_2 = layers.MaxPooling2D(pool_size=(
            column_config['maxPool2'][NUMBER_OF_KERNELS_IDX], column_config['maxPool2'][NUMBER_OF_KERNELS_IDX]),
            padding='valid')(conv_layer_2)

        conv_layer_3 = layers.Conv2D(filters=column_config['conv3'][NUMBER_OF_KERNELS_IDX],
                                     kernel_size=column_config['conv3'][KERNEL_SIZE_IDX],
                                     strides=column_config['conv3'][STRIDE_IDX],
                                     activation='relu',
                                     padding='same')(maxpool_layer_2)

        conv_layer_4 = layers.Conv2D(filters=column_config['conv4'][NUMBER_OF_KERNELS_IDX],
                                     kernel_size=column_config['conv4'][KERNEL_SIZE_IDX],
                                     strides=column_config['conv4'][STRIDE_IDX],
                                     activation='relu',
                                     padding='same')(conv_layer_3)

        deconv_layer_1 = layers.Conv2DTranspose(filters=column_config['conv4'][NUMBER_OF_KERNELS_IDX],
                                                kernel_size=column_config['conv4'][KERNEL_SIZE_IDX],
                                                strides=2,
                                                activation='relu',
                                                padding='same')(conv_layer_4)

        deconv_layer_2 = layers.Conv2DTranspose(filters=column_config['conv3'][NUMBER_OF_KERNELS_IDX],
                                                kernel_size=column_config['conv3'][KERNEL_SIZE_IDX],
                                                strides=2,
                                                activation='relu',
                                                padding='same')(deconv_layer_1)

        self.output = deconv_layer_2

    def get_output(self):
        return self.output


def get_model(image_size):
    inputs = keras.Input(shape=image_size)
    x = layers.Rescaling(1. / 255)(inputs)

    column_1 = IndividualColumn(inputs=x, column_config=config.column1_design).get_output()
    column_2 = IndividualColumn(inputs=x, column_config=config.column2_design).get_output()
    column_3 = IndividualColumn(inputs=x, column_config=config.column3_design).get_output()
    coalesce_columns = layers.Concatenate(axis=3)([column_1, column_2, column_3])
    predict_layer_density_map = layers.Conv2D(filters=config.final_layer_design['conv1'][NUMBER_OF_KERNELS_IDX],
                                              kernel_size=config.final_layer_design['conv1'][KERNEL_SIZE_IDX],
                                              strides=config.final_layer_design['conv1'][STRIDE_IDX],
                                              activation='relu',
                                              padding='same')(coalesce_columns)

    model = keras.Model(inputs, predict_layer_density_map)
    return model


def get_summary(model: keras.Model):
    return model.summary()
