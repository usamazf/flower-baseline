# flower-baseline
A baseline implementation of Federated Learning environment using flower framework. Extending the existing example code provided by flower and modifying it to our needs in order to implement advanced functionalities. Will be extended in our future research.

## Pre-requisites

In order to run this code, you need to install either Pytorch and/or Tensorflow with all their pre-requisites. It is recommended to create a fresh virtual environment before carrying out these installations. 

Best way to do this is with 
[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/overview.html) virtual environment. 

```bash
conda create -n flower_test
conda activate flower_test # some older version of conda require "source activate flower_test" instead.
```

The detailed steps regarding installation of PyTorch and Tensorflow can be found at their respective repositories. [PyTorch ](https://github.com/pytorch/pytorch) and [Tensorflow ](https://github.com/tensorflow/tensorflow).



Finally when you are ready to install flower framework itself. Simply run

```bash
pip install 'flower'
```

## Running the code

Running the code has two distinct parts i.e. starting up the server and initiating the clients. Each of these steps are explained below.

### Staring the Federated Server
First thing we need to do is to run the Federated Server. This can be done by either directly running the ```server.py``` file (with appropriate arguments) located under the ```src``` folder using:

```bash
python server.py \
    --server_address=$SERVER_ADDRESS \
    --rounds=25 \
    --sample_fraction=1.0 \
    --min_sample_size=4 \
    --min_num_clients=4
```

Or we can use the ```run_server.sh``` script located under ```src``` folder and setup appropriate arguments inside the script before hand. 

```bash
./run_server.sh
```

### Starting the Federated Workers

After the server has successfully started next step it to run the Federated Workers. Currently there are two types of workers: the ones that used PyTorch as their DL library and the ones that use Tensorflow as their DL library. We can run either of these clients or a combination of both using the appropriate scripts located under ```src``` folder.

#### Running PyTorch Clients:
```bash
./run_clients_pt.sh
```

#### Running Tensorflow Clients:
```bash
./run_clients_tf.sh
```

## Acknowledgements










