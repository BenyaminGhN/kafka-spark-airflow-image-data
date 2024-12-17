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

BOOTSTRAP_SERVER = "localhost:9092"
TOPIC = "fmri-data"

def configure_kafka():
    # Initialize Kafka consumer
    consumer = KafkaConsumer(
        TOPIC, bootstrap_servers=BOOTSTRAP_SERVER, auto_offset_reset="latest"
    )
    return consumer

def get_model(config):
    model_path = config.model_path

    # Load final model
    final_model = timm.create_model('efficientformerv2_s0.snap_dist_in1k', pretrained=True, num_classes=2)
    model_states = torch.load(final_state_dict, weights_only=True)
    for key in list(model_states.keys()):
        model_states[key.replace('module.', '')] = model_states.pop(key)
    final_model.load_state_dict(model_states)
    final_model.eval()
    
    return model

def get_prediction():
    config_path = Path('.config.yml')
    config = OmegaConf.load(config_path)

    model = get_model(config)
    consumer = configure_kafka()
    print(f"consumer: {consumer}")

    # Process the images in real-time
    for message in consumer:
        print(f"message: {message}")
        message_value = message.value.decode("utf-8")
        data = json.loads(message_value)
        logger.info(f"Consumed message {data["id"]}")

        image_bytes = base64.b64decode(data["image"])
        image = np.frombuffer(image_bytes, dtype=np.float32).reshape(3, 224, 224)
        image = torch.from_numpy(image)
        label = torch.tensor(data["label"])

        print(f"image: {image.shape}")
        print(f"label: {label.shape}")
        # img_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            _, predicted = torch.max(output, 1)

        class_names = ["controlled", "depression"]
        logger.info(f"Classified as: {class_names[predicted.item()]}")

if __name__ == "__main__":
    get_prediction()