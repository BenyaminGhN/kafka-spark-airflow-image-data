from pathlib import Path
import numpy as np
import pandas as pd
import torch
import nibabel as nib
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def load_mri_data(data_dir, participant_id, modality, target_shape):
    file_path = data_dir / str(participant_id) / str(modality) / f'{participant_id}_task-rest_bold.nii.gz'
    if file_path.exists():
        img = nib.load(file_path)
        img_data = img.get_fdata()
        # print(f"Original shape of {participant_id} {modality}: {img_data.shape}")

        mean_fmri_img = img_data.mean(axis=-1)
        # print(f"Mean fMRI image shape: {mean_fmri_img.shape}")

        if len(target_shape) != 3:
            raise ValueError("target_shape should be a tuple of length 3.")

        num_slices = mean_fmri_img.shape[2]
        mid_start = (num_slices - target_shape[2]) // 2
        mid_end = mid_start + target_shape[2]

        mean_fmri_img_middle = mean_fmri_img[:, :, mid_start:mid_end]
        # print(f"Middle slices shape: {mean_fmri_img_middle.shape}")

        zoom_factors = [t / s for t, s in zip(target_shape[:2], mean_fmri_img_middle.shape[:2])]

        mean_fmri_img_resized = zoom(mean_fmri_img_middle, zoom_factors + [1])  # Keep the slice dimension as is
        # print(f"Resized shape of {participant_id} {modality} (mean middle slices): {mean_fmri_img_resized.shape}")

        return mean_fmri_img_resized
    return None

def preporcess(data, labels):
    data_moved_axis = np.moveaxis(data, -1, 1)
    data_reshaped = data_moved_axis.reshape(-1, data_moved_axis.shape[-2], data_moved_axis.shape[-1])

    ## normalizatioin
    data_normalized = (data_reshaped - data_reshaped.min()) / (data_reshaped.max() - data_reshaped.min())
    labels_reshaped = np.repeat(labels, 15)

    data_tensor = torch.tensor(data_normalized, dtype=torch.uint8)
    labels_tensor = torch.tensor(labels_reshaped, dtype=torch.int32)

    data_tensor_u = data_tensor.unsqueeze(1)
    data_tensor_3_channel = data_tensor_u.repeat(1, 3, 1, 1)

    preprocessed_data = data_tensor_3_channel
    preprocessed_labels = labels_tensor

    return preprocessed_data, preprocessed_labels

def get_train_val_generators(config):
    data_dir = Path(config.data_source_path)
    meta_df = pd.read_csv(config.meta_info_path)
    config.meta_info_path

    data =  []
    labels = []
    for index, row in tqdm(meta_df.iterrows(), total=len(meta_df)):
        participant_id = row['participant_id']
        label = row['label']
        mean_fmri_data = load_mri_data(data_dir, participant_id, 'func', config.preprocessing.target_shape)

        if mean_fmri_data is not None:
            data.append(mean_fmri_data)
            labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    prerpocessed_data, preprocessed_labels = preporcess(data, labels)
    train_data, val_data, train_labels, val_labels = train_test_split(
        prerpocessed_data, preprocessed_labels,
        test_size=config.data_pipeline.val_split, random_state=config.seed
    )

    return train_data, val_data, train_labels, val_labels

    # batch_size = config.data_pipeline.batch_size
    # train_dataset = TensorDataset(train_data, train_labels)
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # val_dataset = TensorDataset(val_data, val_labels)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # return train_dataloader, val_dataloader

def get_test_generator(config):
    pass
