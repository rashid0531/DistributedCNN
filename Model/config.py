input_image_width = 224
input_image_height = 224

# These following two values are subjected to be changed based on the CNN architecture.
# For the time being I set it to a mannual value but for future a more efficient way needs to be found out.
# predicted_density_map_width  = 56
# predicted_density_map_height = 56

ground_truth_width = 56
ground_truth_height = 56

# Only greyscale channel was used as the author of the MCNN repository converted the RGB image into greyscale image.
input_image_channels = 3

column1_design = {

            # the parameters for each convolutional layer are arranged as feature_maps/filters,kernel/filter_size,stride
            # and the parameters for each max pool layer are arranged as kernel size followed by strides.
            'conv1' : [16,9,1],
            'maxPool1':[2,2],
            'conv2': [32,7,1],
            'maxPool2': [2,2],
            'conv3': [16,7,1],
            'conv4': [8,7,1],

        }

column2_design = {

            # the parameters for each convolutional layer are arranged as feature_maps/filters,kernel/filter_size,stride
            # and the parameters for each max pool layer are arranged as kernel size followed by strides.
            'conv1' : [20,7,1],
            'maxPool1':[2,2],
            'conv2': [40,5,1],
            'maxPool2': [2,2],
            'conv3': [20,5,1],
            'conv4': [10,5,1],

        }

column3_design = {

            # the parameters for each convolutional layer are arranged as feature_maps/filters,kernel/filter_size,stride
            # and the parameters for each max pool layer are arranged as kernel size followed by strides.
            'conv1' : [24,5,1],
            'maxPool1':[2,2],
            'conv2': [48,3,1],
            'maxPool2': [2,2],
            'conv3': [24,3,1],
            'conv4': [12,3,1],

        }

final_layer_design = {

            # the parameters for each convolutional layer are arranged as feature_maps/filters,kernel/filter_size,stride
            # and the parameters for each max pool layer are arranged as kernel size followed by strides.
            'conv1' : [1,1,1]

        }

