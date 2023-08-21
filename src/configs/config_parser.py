"""Module to parse the configurations provided by user."""

import yaml

def parse_configs(config_path) -> dict:
    """Load and return user configurations."""
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)