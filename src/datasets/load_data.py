#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L I B R A R I E S                                        #
#                                                                            #
#----------------------------------------------------------------------------#

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   return the desired dataset.                                              #
#                                                                            #
#****************************************************************************#
def load_data(dataset_name="cifar-10"):
    
    if dataset_name == "cifar-10":
        # import the appropriate dataset file from the local directory
        from .cifar import load_cifar
        return load_cifar()
    