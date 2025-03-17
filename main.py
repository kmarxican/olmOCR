import base64
import urllib.request

from io import BytesIO
from PIL import Image


import torch
from mlx_vlm import load, apply_chat_template, generate
from mlx_vlm.utils import load_image


#load the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#load the model and processor
olmOCR, model = load("mlx-community/olmOCR-7B-0225-preview-4bit")
olmOCR_processor, processor = load("mlx-community/Qwen2-VL-7B-Instruct") #this might not be necessary at the moment.


