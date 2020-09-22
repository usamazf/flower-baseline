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
    MODEL = "simple-cnn"       # vgg-13, vgg-16, vgg-19, resnet-18, resnet-50, resnet-101, resnet-152
    
    # define the dateset to use for model training
    global DATASET
    DATASET = "cifar-10"           # stl-10, cifar-10, cifar-100, imagenet-1k-64

    # define the id for the created training plan
    global USE_GPU
    USE_GPU = True
    
    # define if quantization should be used
    global QUANTIZE
    QUANTIZE = False
    
    # define the quantization bits [only used if above flag is set to true]
    global Q_BITS
    Q_BITS = 1
    
    # define the batch size you want to use
    #global BATCH_SIZE
    #BATCH_SIZE = 32 
    
    # max batches to process before carrying out the syncrhonization
    #global MAX_NR_BATCHES
    #MAX_NR_BATCHES = 100
    
    # define the total number of epochs you want to train
    #global NUM_EPOCHS
    #NUM_EPOCHS = 10

    # define the initial learning rate to start the training with
    #global INITIAL_LR
    #INITIAL_LR = 1e-3
    
    #-------------------------------------------------------------------------------------------#
    #                                                                                           #
    #   Define process related information to be used by the program.                           #
    #                                                                                           #
    #-------------------------------------------------------------------------------------------#
    
    # define the manual seed for common model initializations.
    global MANUAL_SEED
    MANUAL_SEED = 42        
    
    # other process information about current process.
    #global GLOBAL_RANK
    #GLOBAL_RANK = global_rank
    
    #global LOCAL_RANK
    #LOCAL_RANK = local_rank
    
    #global WORLD_SIZE 
    #WORLD_SIZE = world_size
