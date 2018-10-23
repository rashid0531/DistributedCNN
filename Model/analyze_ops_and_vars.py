# This code is taken from the tf.slim's fithub repo.
# weblink: https://github.com/baidu-research/tensorflow-allreduce/blob/master/tensorflow/contrib/slim/python/slim/model_analyzer.py

def tensor_description(var):
  """Returns a compact and informative string about a tensor.
  Args:
    var: A tensor variable.
  Returns:
    a string with type and size, e.g.: (float32 1x8x8x1024).
  """
  description = '(' + str(var.dtype.name) + ' '
  sizes = var.get_shape()
  for i, size in enumerate(sizes):
    description += str(size)
    if i < len(sizes) - 1:
      description += 'x'
  description += ')'
  return description


def analyze_vars(variables, print_info=False):
  """Prints the names and shapes of the variables.
  Args:
    variables: list of variables, for example tf.global_variables().
    print_info: Optional, if true print variables and their shape.
  Returns:
    (total size of the variables, total bytes of the variables)
  """
  total_size = 0
  total_bytes = 0
  number_of_trainable_vars = 0

  # Arrays to Store variables name, description, dimension and size in bytes.
  var_names = []
  var_descriptions = []
  var_dimensions = []
  var_bytes_arr = []

  for var in variables:
    number_of_trainable_vars += 1  
    # if var.num_elements() is None or [] assume size 0.
    var_size = var.get_shape().num_elements() or 0
    var_bytes = var_size * var.dtype.size
    total_size += var_size
    total_bytes += var_bytes

    # store values in corresponding arrays.
    var_names.append(var.name)
    var_descriptions.append(tensor_description(var))
    var_dimensions.append(var_size)
    var_bytes_arr.append(var_bytes)

    if print_info:
      print(var.name, tensor_description(var), '[%d, bytes: %d]' %
            (var_size, var_bytes))

  print('Total number of variables: %d' % number_of_trainable_vars)  
  print('Total size of variables: %d' % total_size)
  print('Total bytes of variables: %d' % total_bytes)

  with open("DataflowGraph.txt","w") as file_obj:

      for i in range(0,len(variables)):
          format_str = ('variable name %s, variable description %s, variable dimension %s, variable size in bytes %s \n')
          informations = format_str % (var_names[i], var_descriptions[i], var_dimensions[i], var_bytes_arr[i])
          file_obj.write(informations)

  #return total_size, total_bytes


