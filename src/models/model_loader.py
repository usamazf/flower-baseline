"""A function to load desired model for training."""

def load_model(model_configs: dict):
    
    assert model_configs["MODEL_NAME"] in ["SIMPLE-CNN", "SIMPLE-MLP"], f"Invalid model {model_configs['MODEL_NAME']} requested."

    if model_configs["MODEL_NAME"] == "SIMPLE-MLP":
        from .simple_mlp import Net
        return Net(num_classes = model_configs["NUM_CLASSES"])
    elif model_configs["MODEL_NAME"] == "SIMPLE-CNN":
        from .simple_cnn import Net
        return Net(num_classes = model_configs["NUM_CLASSES"])
    else:
        raise Exception(f"Invalid model {model_configs['MODEL_NAME']} requested.")
