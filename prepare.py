from pathlib import Path
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from src.data_ingestion import create_meta_df
from src.data_preparation import get_train_val_generators
from omegaconf import OmegaConf

import torchvision

def main():
    config_path = Path("config.yml")
    config = OmegaConf.load(config_path)

    meta_df_path = create_meta_df(config)
    meta_df = pd.read_csv(meta_df_path)
    print(meta_df)
    print(meta_df["label"].values)

    train_gen, val_gen = get_train_val_generators(config)
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig("asset/data.png", dpi=300)
        plt.show()

    # get some random training images
    dataiter = iter(train_gen)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(config.data_pipeline.batch_size)))

if __name__ == "__main__":
    main()