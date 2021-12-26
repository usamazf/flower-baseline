#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L I B R A R I E S                                        #
#                                                                            #
#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
#                                                                            #
#   Define global parameters and initialize them to default values.          #
#                                                                            #
#----------------------------------------------------------------------------#
def init( ): # setup default configurations
        
    # define the model to use for training
    global MODEL
    MODEL = "simple-cnn"         # simple-cnn, mlp-mnist
    
    # define the dateset to use for model training
    global DATASET
    DATASET = "cifar-10"           # mnist, cifar-10

    # define the id for the created training plan
    global USE_GPU
    USE_GPU = False
    
    # define if quantization should be used
    global QUANTIZE
    QUANTIZE = False
    
    # define the quantization bits [only used if above flag is set to true]
    global Q_BITS
    Q_BITS = 16
        
    #-------------------------------------------------------------------------------------------#
    #                                                                                           #
    #   Define process related information to be used by the program.                           #
    #                                                                                           #
    #-------------------------------------------------------------------------------------------#
    
    # define the manual seed for common model initializations.
    global MANUAL_SEED
    MANUAL_SEED = 42        

#----------------------------------------------------------------------------#
#                                                                            #
#   Setup global parameters according to the provided values.                #
#                                                                            #
#----------------------------------------------------------------------------#
def setup(model, dataset, use_gpu, quantize, q_bits, manual_seed):   
    global MODEL
    MODEL = model
    
    global DATASET
    DATASET = dataset

    global USE_GPU
    USE_GPU = use_gpu
    
    global QUANTIZE
    QUANTIZE = quantize
    
    global Q_BITS
    Q_BITS = q_bits

    global MANUAL_SEED
    MANUAL_SEED = manual_seed
