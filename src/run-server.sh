#!/bin/bash

SERVER_ADDRESS="12.12.12.101:8888" #"192.168.15.91:8888"

# Start a Flower server
python server.py \
    --server_address=$SERVER_ADDRESS \
    --rounds=25 \
    --sample_fraction=1.0 \
    --min_sample_size=4 \
    --min_num_clients=4
