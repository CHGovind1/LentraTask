import pandas as pd
import io
import requests
import yaml
import logging

def load_config():
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def ingest_data():
    """Download German Credit data from UCI"""
    config = load_config()
    
    logging.info("Downloading data from %s", config["data"]["url"])
    response = requests.get(config["data"]["url"])
    data = response.text.strip().split("\n")
    
    df = pd.read_csv(
        io.StringIO("\n".join(data)),
        sep=" ",
        header=None,
        names=config["data"]["columns"]
    )
    logging.info(f"Loaded {len(df)} rows")
    return df

