import os
import json
import pandas as pd
from src.utils.logging import setup_logger

logger = setup_logger(__name__)

def load_raw_dataset(raw_dir):
    """Finds and loads the primary raw dataset from raw_dir."""
    logger.info(f"Scanning for raw datasets in: {raw_dir}")
    for fname in sorted(os.listdir(raw_dir)):
        fpath = os.path.join(raw_dir, fname)
        if fname.startswith(".") or os.path.isdir(fpath):
            continue
        
        if fname.endswith(".json"):
            with open(fpath, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            logger.info(f"Loaded JSON raw dataset: {fname} (Shape: {df.shape})")
            return df
        elif fname.endswith(".csv"):
            df = pd.read_csv(fpath)
            logger.info(f"Loaded CSV raw dataset: {fname} (Shape: {df.shape})")
            return df
        elif fname.endswith((".xlsx", ".xlsm")):
            df = pd.read_excel(fpath, engine="openpyxl")
            logger.info(f"Loaded Excel raw dataset: {fname} (Shape: {df.shape})")
            return df
            
    raise FileNotFoundError(f"No suitable raw dataset file found in {raw_dir}")
