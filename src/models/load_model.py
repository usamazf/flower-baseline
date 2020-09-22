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
def load_model(model_name="simple-cnn", num_classes=10, framework="PT"):
    
    # Load the required model for required framework
    if model_name == "simple-cnn":
        # import the model from the local directory
        if framework == "PT":
            from .pt_models.simple_cnn import Net
            return Net()
        elif framework == "TF":
            from .tf_models.simple_cnn import Net
            return Net()
    
    elif model_name == "SOME OTHER":
        pass
