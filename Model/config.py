# Shape of each training image.
input_image_width = 224
input_image_height = 224

'''
The parameters for each convolutional layer are arranged as feature_maps/filters,kernel/filter_size,stride 
and the parameters for each max pool layer are arranged as kernel size followed by strides.
'''
column1_design = { 
    'conv1' : [16,9,1],
    'maxPool1':[2,2],
    'conv2': [32,7,1],
    'maxPool2': [2,2],
    'conv3': [16,7,1],
    'conv4': [8,7,1],
}

column2_design = {
    'conv1' : [20,7,1],
    'maxPool1':[2,2],
    'conv2': [40,5,1],
    'maxPool2': [2,2],
    'conv3': [20,5,1],
    'conv4': [10,5,1],
}

column3_design = {
    'conv1' : [24,5,1],
    'maxPool1':[2,2],
    'conv2': [48,3,1],
    'maxPool2': [2,2],
    'conv3': [24,3,1],
    'conv4': [12,3,1],
}

final_layer_design = {
    'conv1' : [1,1,1]
    }

