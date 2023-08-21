"""A function to load and split the desired dataset among clients."""

def load_data(dataset_name: str, 
              dataset_path: str):
    
    assert dataset_name in ["MNIST", "CIFAR-10", "EMNIST"], f"Invalid dataset {dataset_name} requested."

    if dataset_name == "MNIST":
        from .dt_mnist import load_mnist
        return load_mnist(data_root=dataset_path)
    elif dataset_name == "EMNIST":
        from .dt_emnist import load_emnist
        return load_emnist(data_root=dataset_path)
    elif dataset_name == "CIFAR-10":
        from .dt_cifar10 import load_cifar10
        return load_cifar10(data_root=dataset_path)
    else:
        raise Exception(f"Invalid dataset {dataset_name} requested.")


def load_and_fetch_split(
        client_id: int,
        n_clients: int,
        dataset_conf: dict
    ):
    """A routine to load and split data."""

    # load the dataset requested
    trainset, testset \
        = load_data(dataset_name=dataset_conf["DATASET_NAME"],
                    dataset_path=dataset_conf["DATASET_PATH"]
                   )

    # split the dataset if requested
    if dataset_conf["SPLIT"]:
        from .data_split import split_data
        split_train, split_labels \
            = split_data(train_data = trainset, 
                         dirichlet_alpha = dataset_conf["DIRICHLET_ALPHA"], 
                         client_id = client_id,
                         n_clients = n_clients,
                         random_seed = dataset_conf["RANDOM_SEED"]
                        )
        return (split_train, split_labels), testset
    else:
        return (trainset, None), testset