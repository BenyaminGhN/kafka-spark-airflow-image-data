import base64
import json
import logging
import sys
import time
import types
from uuid import uuid4
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Following session is to fix the six.moves issue in kafka-python package
m = types.ModuleType("kafka.vendor.six.moves", "Mock module")
setattr(m, "range", range)
sys.modules["kafka.vendor.six.moves"] = m

from kafka import KafkaProducer
from confluent_kafka import Producer
from torch.utils.data import DataLoader, TensorDataset

from src.data_preparation import load_mri_data, preporcess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = ['broker:9092']
# BOOTSTRAP_SERVER = "localhost:9092"
TOPIC = "fmri-data"

def configure_kafka(servers=KAFKA_BOOTSTRAP_SERVERS):
    """Creates and returns a Kafka producer instance."""
    settings = {
        'bootstrap.servers': ','.join(servers),
        # 'client.id': 'producer_instance'  
    }
    return Producer(settings)

def get_raw_data(config):
    data_dir = config.test_data_dir
    return data_dir

def prepare_data(config):
    data_dir = Path(config.test_data_dir)
    data_path = [x for x in data_dir.rglob('*') if x.is_dir() and ("sub" in str(x.name))]

    case_id_list = [str(x.name) for x in data_path]
    data =  []
    labels = []
    for case_id in tqdm(case_id_list, total=len(case_id_list)):
        participant_id = case_id
        label = 0

        mean_fmri_data = load_mri_data(data_dir, participant_id, 'func', config.preprocessing.target_shape)

        if mean_fmri_data is not None:
            data.append(mean_fmri_data)
            labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    prerpocessed_data, preprocessed_labels = preporcess(data, labels)
    dataset = TensorDataset(prerpocessed_data, preprocessed_labels)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    return data_loader

def delivery_status(err, msg):
    """Reports the delivery status of the message to Kafka."""
    if err is not None:
        print('Message delivery failed:', err)
    else:
        print('Message delivered to', msg.topic(), '[Partition: {}]'.format(msg.partition()))

def send_image(image, label, producer):
    # Convert image to bytes and then to base64 string
    image_bytes = image.numpy().tobytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Create a message with image and label
    message = {"image": image_base64, "label": label.item(), "id": str(uuid4())}

    # Send the message to Kafka
    producer.produce(TOPIC, json.dumps(message).encode("utf-8"), on_delivery=delivery_status)
    producer.flush()
    logger.info("Produced image")

def produce():
    config_path = Path("./config.yml")
    config = OmegaConf.load(config_path)

    producer = configure_kafka()
    raw_data_path = get_raw_data(config)
    data_loader = prepare_data(config)
    for i, data in enumerate(data_loader, 0):
        images, labels = data
        print(images.shape)
        send_image(images[0], labels[0], producer)
        time.sleep(5)  # Send messages with 5 second delay

    logger.info("Finished sending images to Kafka")

if __name__ == "__main__":
    produce()