import pandas as pd 
from pathlib import Path
import numpy as np
import json
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


def read_meta_info():
    base_dir = Path("./data/data-source")
    participants_file = base_dir / 'participants.tsv'
    participants_df = pd.read_csv(participants_file, sep='\t')

    return participants_df

def create_meta_df(config, participants_df, path_to_csv="./data/meta_info.csv"):
    meta_data = []
    for participant_id in participants_df['participant_id']:
        anat_path = os.path.join(base_dir, participant_id, 'anat', f'{participant_id}_T1w.nii.gz')
        func_path = os.path.join(base_dir, participant_id, 'func', f'{participant_id}_task-rest_bold.nii.gz')

        if os.path.exists(anat_path) and os.path.exists(func_path):
            participant_data = participants_df[participants_df['participant_id'] == participant_id].to_dict(orient='records')[0]
            participant_data['anat_path'] = anat_path
            participant_data['func_path'] = func_path

            # add label
            participant_data['label'] = 1 if participant_data['group'].values == 'dper' else 0

            meta_data.append(participant_data)

    meta_df = pd.DataFrame(meta_data)

    splits = np.array(['train']*len(meta_df), dtype='object')
    eval_split = config.data_ingestion.eval_split
    eval_indices = np.random.RandomState(config.seed).randint(0, len(splits), int(len(splits) * eval_split))
    splits[eval_indices] = 'evaluation'
    meta_df['Split'] = splits

    if path_to_csv is not None:
        meta_df.to_csv(path_to_csv, index=False)

    return path_to_csv
    
