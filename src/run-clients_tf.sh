#!/bin/bash

SERVER_ADDRESS="12.12.12.101:8888" #"192.168.15.91:8888"

NUM_CLIENTS=2
I_START=0
I_END=1

echo "Starting $NUM_CLIENTS clients."
for ((i = $I_START; i <= $I_END; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    python client_tf.py \
      --cid=$i \
      --server_address=$SERVER_ADDRESS &
done
echo "Started $NUM_CLIENTS clients."
