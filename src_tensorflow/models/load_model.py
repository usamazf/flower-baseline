#----------------------------------------------------------------------------#
#                                                                            #
#   I M P O R T     L I B R A R I E S                                        #
#                                                                            #
#----------------------------------------------------------------------------#

#****************************************************************************#
#                                                                            #
#   description:                                                             #
#   return the desired model.                                                #
#                                                                            #
#****************************************************************************#
def load_model(model_name="fashion-mnist", num_classes=10):
    
    
    if model_name == "fashion-mnist":
        # import the model from the local directory
        from .pyt_cnn import Net
        return Net()
    
    #elif model_name == "mnist-small":
    #    return Net()
    
    #elif model_name == "vgg-13":
    #    return vgg13_model(nClasses=num_classes)
    
    #elif model_name == "vgg-16":
    #    return vgg16_model(nClasses=num_classes)
    
    #elif model_name == "vgg-19":
    #    return vgg19_model(nClasses=num_classes)