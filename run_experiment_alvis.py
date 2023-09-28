from os import listdir
from os.path import isfile, join
import subprocess
import argparse
import re
import math
from src.configs import parse_configs

def main():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--num_gpus",
        type=int,
        required=True,
        help="Number of Allocated GPUs (no default)",
    )
    parser.add_argument(
        "--configs_path",
        type=str,
        required=True,
        help="Configuration file path (no default)",
    )
    args = parser.parse_args()
    print(f"# of GPUs    : {args.num_gpus}")
    print(f"Configs Path : {args.configs_path}")
    all_configs = [join(args.configs_path, f) for f in listdir(args.configs_path) if isfile(join(args.configs_path, f))]

    for config_file in all_configs:
        print(f"Running experiments for: {config_file}")
        user_configs = parse_configs(config_file)
        total_client = user_configs["SERVER_CONFIGS"]["MIN_NUM_CLIENTS"]
        server_call = f'python src/run_fl_server.py --server_address="0.0.0.0:59999" --config_file="{config_file}"'
        server_proc = subprocess.Popen([server_call], shell=True)
        client_call = f'python src/run_fl_clients.py --server_address="127.0.0.1:59999" --max_gpus={args.num_gpus} --total_clients={total_client} --num_clients={total_client} --start_cid=0 --config_file="{config_file}"'
        client_proc = subprocess.Popen([client_call], shell=True)
        # Wait for processes to terminate
        client_proc.wait()
        server_proc.wait()

if __name__=="__main__":
    main()
