#!/bin/bash

SERVER_ADDRESS="192.168.15.73:8888"

# Start a Flower server
python server.py \
    --server_address=$SERVER_ADDRESS \
    --rounds=5 \
    --sample_fraction=1.0 \
    --min_sample_size=2 \
    --min_num_clients=2
