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
def load_data(dataset_name="cifar-10", framework="PT"):
    
    if dataset_name == "cifar-10":
        # import the appropriate dataset file from the local directory
        if framework == "PT":
            from .pt_datasets.cifar import load_cifar
            return load_cifar()
        elif framework == "TF":
            from .tf_datasets.cifar import load_cifar
            return load_cifar()
    
    elif dataset_name == "mnist":
        # import the appropriate dataset file from the local directory
        if framework == "PT":
            from .pt_datasets.mnist import load_mnist
            return load_mnist()
        elif framework == "TF":
            from .tf_datasets.mnist import load_mnist
            return load_mnist()

    elif dataset_name == "SOMETHING ELSE":
        pass
    