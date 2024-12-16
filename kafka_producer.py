import base64
import json
import logging
import sys
import time
import types
from uuid import uuid4

# Following session is to fix the six.moves issue in kafka-python package
m = types.ModuleType("kafka.vendor.six.moves", "Mock module")
setattr(m, "range", range)
sys.modules["kafka.vendor.six.moves"] = m

from kafka import KafkaProducer
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DATA_PATH = "./data"
BOOTSTRAP_SERVER = "localhost:9092"
TOPIC = "fmri-data"

# Set up CIFAR-10 dataset
transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=1, shuffle=True)

# Set up Kafka producer
producer = KafkaProducer(bootstrap_servers=[BOOTSTRAP_SERVER])

def send_image(image, label):
    # Convert image to bytes and then to base64 string
    image_bytes = image.numpy().tobytes()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # Create a message with image and label
    message = {"image": image_base64, "label": label.item(), "id": str(uuid4())}

    # Send the message to Kafka
    producer.send(TOPIC, json.dumps(message).encode("utf-8"))
    producer.flush()
    logger.info("Produced image")

def intiate_stream():
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        send_image(images[0], labels[0])
        time.sleep(5)  # Send messages with 5 second delay

    logger.info("Finished sending images to Kafka")

if __name__ == "__main__":
    intiate_stream()