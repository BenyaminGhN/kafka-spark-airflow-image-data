from pathlib import Path
import os
import torch
import timm
import torchmetrics
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torchvision.transforms as transforms
from omegaconf import OmegaConf
import timm
from pyspark.ml.torch.distributor import TorchDistributor
from src.data_ingestion import create_meta_df
from src.data_preparation import get_train_val_generators

from pyspark.sql import SparkSession
from pyspark import SparkContext
# SparkSession.builder.appName("torch-training").getOrCreate()

def train(train_dataset, val_dataset, device):

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)

    model = timm.create_model('efficientformerv2_s0.snap_dist_in1k', pretrained=True, num_classes=2).to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    accuracy_metric = torchmetrics.Accuracy(task='binary').to(device)

    sampler = DistributedSampler(train_dataset, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=sampler)

    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model.train()
    for epoch in range(2):
        loop = tqdm(train_dataloader, leave=False)

        model.train()
        for batch in loop:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)

            loss = torch.nn.functional.cross_entropy(outputs, labels)

            loss.backward()
            optimizer.zero_grad()

            # # Gradient aggregation using allreduce
            # for param in model.parameters():
            #     if param.grad is not None:
            #         torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            #         param.grad.data /= world_size

            optimizer.step()
            accuracy_metric(predictions, labels)

            loop.set_description(f'Epoch {epoch + 1} [Train]')
            loop.set_postfix(loss=loss.item())

        train_accuracy = accuracy_metric.compute()
        accuracy_metric.reset()

        model.eval()
        val_accuracy_metric = torchmetrics.Accuracy(task='binary').to(device)
        val_loss_total = 0

        with torch.no_grad():
            for val_batch in val_dataloader:
                val_inputs, val_labels = val_batch
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)
                val_predictions = val_outputs.argmax(dim=1)

                val_loss = torch.nn.functional.cross_entropy(val_outputs, val_labels)
                val_loss_total += val_loss.item()

                val_accuracy_metric(val_predictions, val_labels)

        val_accuracy = val_accuracy_metric.compute()
        val_accuracy_metric.reset()

        print(f"Epoch {epoch + 1}: Train Accuracy = {train_accuracy:.4f}, Val Accuracy = {val_accuracy:.4f}, Val Loss = {val_loss_total / len(val_dataloader):.4f}")
    
    torch.distributed.destroy_process_group()
    torch.save(model.state_dict(), "./model/vit/model.pth.tar")
    return model.state_dict()

def main():
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("PyTorch Distributed Training with PySpark") \
        .getOrCreate()

    sc = spark.sparkContext

    config_path = Path("config.yml")
    config = OmegaConf.load(config_path)

    ## create meta_df
    meta_df_path = create_meta_df(config)

    ## prepare_data
    # train_dataloader, val_dataloader = get_train_val_generators(config)
    train_data, val_data, train_labels, val_labels = get_train_val_generators(config)
    batch_size = config.data_pipeline.batch_size
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(os.getenv('OMP_NUM_THREADS'))

    # Parallelize data
    num_procc = 2
    # rdd = sc.parallelize(list(range(num_procc)), numSlices=num_procc)
    # print(rdd.collect())
    # rdd = sc.parallelize(train_dataloader, numSlices=4)

    distributor = TorchDistributor(
        num_processes=num_procc,
        local_mode=True,
        use_gpu=False
    )

    model_states = distributor.run(train, train_dataset, val_dataset, device)
    # torch.save(model, config.model_path)

    # # Aggregate model weights
    # final_state_dict = model_states[0]
    # for key in final_state_dict.keys():
    #     final_state_dict[key] = torch.stack([state[key] for state in model_states]).mean(0)

    # Load final model

    final_model = timm.create_model('efficientformerv2_s0.snap_dist_in1k', pretrained=False, num_classes=2)
    model_states = torch.load(config.model_path, weights_only=True)
    print(model_states.keys())
    for key in list(model_states.keys()):
        model_states[key.replace('module.', '')] = model_states.pop(key)
    final_model.load_state_dict(model_states)

    print("Training completed. Final model state_dict:")
    
if __name__ == "__main__":
    main()