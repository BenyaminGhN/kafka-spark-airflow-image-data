import base64
import json
import logging
import sys
import types
from omegaconf import OmegaConf

# Following session is to fix the six.moves issue in kafka-python package
m = types.ModuleType("kafka.vendor.six.moves", "Mock module")
setattr(m, "range", range)
sys.modules["kafka.vendor.six.moves"] = m

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils
import torchvision.transforms as transforms
from kafka import KafkaConsumer
import timm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "./cifar_net.pth"
DATA_PATH = "./data"
BOOTSTRAP_SERVER = "localhost:9092"
TOPIC = "fmri-data"

# Initialize Kafka consumer
consumer = KafkaConsumer(
    TOPIC, bootstrap_servers=BOOTSTRAP_SERVER, auto_offset_reset="latest"
)

def get_model(config):
    model_path = "./model/"

    # Load final model
    final_model = timm.create_model('efficientformerv2_s0.snap_dist_in1k', pretrained=True, num_classes=2)
    final_model.load_state_dict(torch.load(final_state_dict))
    final_model.eval()
    
    return model

# Pre-processing function
preprocess = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),  # Assuming the model expects 32x32 images
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

def get_prediction():
    # Process the images in real-time
    for message in consumer:
        message_value = message.value.decode("utf-8")
        data = json.loads(message_value)
        logger.info(f"Consumed message {data["id"]}")

        image_bytes = base64.b64decode(data["image"])

        prepared_image = get
        image = np.frombuffer(image_bytes, dtype=np.float32).reshape(3, 32, 32)
        # imshow(image)
        image = torch.from_numpy(image)
        label = torch.tensor(data["label"])

        img_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)

        logger.info(f"Classified as: {class_names[predicted.item()]}")

if __name__ == "__main__":
    get_prediction()