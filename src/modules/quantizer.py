#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L I B R A R I E S                                        #
#                                                                            #
#----------------------------------------------------------------------------#
import timeit

import numpy as np
import flwr as fl

#----------------------------------------------------------------------------#
#                                                                            #
#   Define global parameters to be used through out this module              #
#                                                                            #
#----------------------------------------------------------------------------#
BT_COL = 64
DT_TYP = np.uint32 if BT_COL==32 else np.uint64

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   function used to perform actual bit quantization.                        #
#                                                                            #
#****************************************************************************#
def bit_quantization(weights, bits):
    # get min and max values
    min_value = weights.min()
    max_value = weights.max()
    # create bins / intervals for quantization
    levels = np.linspace(min_value, max_value, num = 2**bits)
    # create the bit format
    bit_format = '{:0' + str(bits) + 'b}'
    # find nearest values for the given weights array
    bit_representation = [
        bit_format.format((np.abs(levels-value)).argmin()) for value in weights
    ]
    # return the results
    return min_value, max_value, bit_representation

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   function used to perform actual bit dequantization.                      #
#                                                                            #
#****************************************************************************#
def bit_dequantization(min_value, max_value, bit_repr, bits):
    levels = np.linspace(min_value, max_value, num = 2**bits)
    dequantized = [levels[int(bit_repr[i:i+bits], 2)] for i in range(0, len(bit_repr), bits)]
    return dequantized

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   internal function used to compute chunks sizes for dequantized array.    #
#                                                                            #
#****************************************************************************#
def get_reshape_params(dummy_model):
    count = 0
    param_index = []
    layer_shape = []
    # check layer parameters count
    params = dummy_model.get_weights()
    # get required results
    for val in params:
        layer_shape.append(val.shape)
        param_index.append(count + val.size)
        count += val.size
    # return layer shapes and slit index
    return layer_shape, param_index

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   function used to perform quantization and other required steps.          #
#                                                                            #
#****************************************************************************#
def quantize(weights: fl.common.Weights, bits: int=1) -> fl.common.Weights:
    
    # time the quantization process
    quant_begin = timeit.default_timer()
    
    # flatten all the numpy arrays
    flat_weights = [np_array.flatten() for np_array in weights]

    # stitch flattened arrays to a single array
    weights_vector = np.concatenate(flat_weights)
    
    # perform actual qunatization
    min_value, max_value, bit_vector = bit_quantization(weights_vector, bits)
    
    # perform post-processing
    binary_rep = ''.join(bit_vector)
    len_binary = len(binary_rep)
    
    # create the bit format
    bit_format = '{:0>'+str(BT_COL)+'}'
    
    # convert the bits to a number representation (32 / 64 bits at a time)
    numbers = []
    for i in range(0, len(binary_rep), BT_COL):
        bin_N_bit = bit_format.format(binary_rep[i:i+BT_COL])
        numbers.append(int(bin_N_bit, 2))
    
    # time the method execution
    quant_duration = timeit.default_timer() - quant_begin
    
    # create numpy ndarray to return results
    npArray = np.array(numbers, dtype=DT_TYP)
    
    # print the processing time to console
    print("Quantization process took:", quant_duration)
    
    # finally return all results for transmission
    return [np.array([min_value, max_value, len_binary]), npArray]

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   function used to perform actual bit dequantization.                      #
#                                                                            #
#****************************************************************************#
def dequantize(dummy_model, weights: fl.common.Weights, bits: int=1) -> fl.common.Weights:
    
    # time the dequantization process
    dequant_begin = timeit.default_timer()
    
    # expand the values
    min_value = float(weights[0][0])
    max_value = float(weights[0][1])
    len_binary = int(weights[0][2])
    numbers = weights[1] # the actual quantized values
    
    # create the bit format
    bit_format = '{:0'+str(BT_COL)+'b}'
    
    # process the number
    bit_N_vector = [bit_format.format(number) for number in numbers]
    binary_rep = ''.join(bit_N_vector)
    
    # do bit dequantization
    weight_vector = bit_dequantization(min_value, max_value, binary_rep, bits)
    
    # reshaping the weight vector to desired model
    layer_shapes, param_index = get_reshape_params(dummy_model)
    split_vectors = np.split(weight_vector, param_index)[:-1]
    
    weights: fl.common.Weights = [
        layer_params.reshape(layer_shapes[indx]) for indx, layer_params in enumerate(split_vectors)
    ]
    
    # time the method execution
    dequant_duration = timeit.default_timer() - dequant_begin
    
    print("Dequantization process took:", dequant_duration)
    
    # finally return the dequantized weights
    return weights
