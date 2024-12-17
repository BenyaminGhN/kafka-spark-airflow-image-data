import pandas as pd 
from pathlib import Path
import numpy as np
import json
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import shutil


def create_meta_df(config, to_save=True):
    data_dir = Path(config.data_source_path)
    participants_df = pd.read_csv(data_dir / "participants.tsv", sep='\t')
    meta_data = []
    for participant_id in participants_df['participant_id']:
        anat_path = data_dir / str(participant_id) / 'anat' / f'{participant_id}_T1w.nii.gz'
        func_path = data_dir / str(participant_id) / 'func' / f'{participant_id}_task-rest_bold.nii.gz'

        if anat_path.exists() and func_path.exists():
            participant_data = participants_df[participants_df['participant_id'] == participant_id].to_dict(orient='records')[0]
            participant_data['anat_path'] = str(anat_path)
            participant_data['func_path'] = str(func_path)

            # add label
            participant_data['label'] = 1 if str(participant_data['group']) == "depr" else 0

            meta_data.append(participant_data)

    meta_df = pd.DataFrame(meta_data)

    splits = np.array(['train']*len(meta_df), dtype='object')
    eval_split = config.data_ingestion.eval_split
    eval_indices = np.random.RandomState(config.seed).randint(0, len(splits), int(len(splits) * eval_split))
    splits[eval_indices] = 'evaluation'
    meta_df['Split'] = splits

    if to_save:
        meta_df.to_csv(config.meta_info_path, index=False)

    return config.meta_info_path

def create_datalake_test():
    config_path = Path("./config.yml")
    config = OmegaConf.load(config_path)

    data_dir = Path(config.data_source_path)
    meta_df = pd.read_csv(config.meta_info_path)
    val_df = meta_df[meta_df["Split"]=="evaluation"]
    sub_df = val_df.sample(4, random_state=123)
    for participant_id in sub_df['participant_id']:
        case_dir = data_dir / str(participant_id) 
        source = str(case_dir)
        destination = str(Path(config.test_data_dir) / case_dir.name)
        shutil.copytree(source, destination, dirs_exist_ok = True)

if __name__ == "__main__":
    pass    