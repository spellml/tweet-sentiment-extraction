# import torch
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
        pass

    def predict(self, payload):
        return "Hello World!"
