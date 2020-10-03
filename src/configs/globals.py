#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L I B R A R I E S                                        #
#                                                                            #
#----------------------------------------------------------------------------#

#----------------------------------------------------------------------------#
#                                                                            #
#   Define global parameters to be used through out the program              #
#                                                                            #
#----------------------------------------------------------------------------#
def init( ):
        
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
    Q_BITS = 1
        
    #-------------------------------------------------------------------------------------------#
    #                                                                                           #
    #   Define process related information to be used by the program.                           #
    #                                                                                           #
    #-------------------------------------------------------------------------------------------#
    
    # define the manual seed for common model initializations.
    global MANUAL_SEED
    MANUAL_SEED = 42        

