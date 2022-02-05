import tensorflow as tf
import config as config

class MCNN():

    def __init__(self,input_image):

        # column1_design : a tuple that contains the parameters for each layer of CNN.

        self._column1_design = config.column1_design
        self._column2_design = config.column2_design
        self._column3_design = config.column3_design
        self._final_layer_design = config.final_layer_design

        self.column1_output = self.Shallow(input_image,self._column1_design, 'column1/')
        self.column2_output = self.Shallow(input_image,self._column2_design, 'column2/')
        self.column3_output = self.Shallow(input_image,self._column3_design, 'column3/')
        self.fusion = tf.concat([self.column1_output,self.column2_output,self.column3_output],axis = 3)
        self.final_layer_output = self.final_layer(self.fusion,self._final_layer_design)


    def Shallow(self,input_image,properties,variable_layer_name):

        # First convolutional layer
        conv1 = tf.layers.conv2d(input_image, filters=properties['conv1'][0], kernel_size=properties['conv1'][1],
                                 strides=[properties['conv1'][2], properties['conv1'][2]], padding="SAME", activation=tf.nn.relu,name = variable_layer_name +'conv1')

        # Max pool layer - 1st
        max_pool1 = tf.nn.max_pool(conv1, ksize=[1, properties['maxPool1'][0], properties['maxPool1'][0], 1],
                                   strides=[1, properties['maxPool1'][1],properties['maxPool1'][1], 1], padding="VALID")

        # 2nd convolutional layer
        conv2 = tf.layers.conv2d(max_pool1, filters=properties['conv2'][0], kernel_size=properties['conv2'][1],
                                 strides=[properties['conv2'][2], properties['conv2'][2]], padding="SAME",
                                 activation=tf.nn.relu,name = variable_layer_name +'conv2')

        # Max pool layer - 2nd
        max_pool2 = tf.nn.max_pool(conv2, ksize=[1, properties['maxPool2'][0], properties['maxPool2'][0], 1],
                                   strides=[1, properties['maxPool2'][1], properties['maxPool2'][1], 1],
                                   padding="VALID")

        # 3rd convolutional layer
        conv3 = tf.layers.conv2d(max_pool2, filters=properties['conv3'][0], kernel_size=properties['conv3'][1],
                                 strides=[properties['conv3'][2], properties['conv3'][2]], padding="SAME",
                                 activation=tf.nn.relu,name = variable_layer_name + 'conv3')

        # 3rd convolutional layer
        conv4 = tf.layers.conv2d(conv3, filters=properties['conv4'][0], kernel_size=properties['conv4'][1],
                                 strides=[properties['conv4'][2], properties['conv4'][2]], padding="SAME",
                                 activation=tf.nn.relu,name = variable_layer_name + 'conv4')

        # 1st Deconvolutional layer
        transposed_conv1 = tf.layers.conv2d_transpose(conv4,filters=properties['conv4'][0], kernel_size=properties['conv4'][1],
                                 strides=[2,2], padding="SAME",
                                 activation=tf.nn.relu,name = variable_layer_name + 'deconv1')

        # 2nd Deconvolutional layer
        transposed_conv2 = tf.layers.conv2d_transpose(transposed_conv1,filters=properties['conv3'][0], kernel_size=properties['conv3'][1], 
                                 strides=[2,2], padding="SAME",
                                 activation=tf.nn.relu,name = variable_layer_name + 'deconv2')


        return transposed_conv2

    def final_layer(self,input,properties):

        final_conv = tf.layers.conv2d(input, filters=properties['conv1'][0], kernel_size=properties['conv1'][1],
                                 strides=[properties['conv1'][2], properties['conv1'][2]], padding="SAME",
                                 activation=tf.nn.relu,name = 'final_conv')

        return final_conv