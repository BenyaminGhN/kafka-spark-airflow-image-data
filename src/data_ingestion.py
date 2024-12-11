import pandas as pd 
from pathlib import Path
import numpy as np
import json

def read_csv_info():
    base_dir = Path("./data")
    participants_file = base_dir / 'participants.tsv'
    participants_df = pd.read_csv(participants_file, sep='\t')