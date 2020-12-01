import torch
# import numpy as np
# from PIL import Image

# import os
# import yaml
# import io
# import base64
# import json

from spell.serving import BasePredictor

class Predictor(BasePredictor):
    def __init__(self):
        if torch.cuda.device_count() >= 1:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = torch.load(f"model/model.pth", map_location=self.device)

    def predict(self, payload):
        return payload
