import torch
from torch import autocast
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline

import numpy as np
from spectro import wav_bytes_from_spectrogram_image

import base64
from io import BytesIO
import os

def dummy_checker(images, **kwargs): return images, False

def init():
    global pipe
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    repo = 'spaceinvader/fb'
#     scheduler = DPMSolverMultistepScheduler.from_pretrained(repo, subfolder="scheduler")
    device = "cuda"
    MODEL_ID = "spaceinvader/fb"
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    pipe = pipe.to(device)
#     model = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.float16, revision="main", scheduler=scheduler, use_auth_token=HF_AUTH_TOKEN).to("cuda")    

def inference(model_inputs:dict):
    global pipe

    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 512)
    width = model_inputs.get('width', 512)
    steps = model_inputs.get('steps', 20)
    guidance_scale = model_inputs.get('guidance_scale', 9)
    seed = model_inputs.get('seed', None)

    if not prompt: return {'message': 'No prompt was provided'}
    
    generator = None
    if seed: generator = torch.Generator("cuda").manual_seed(seed)
    
    pipe.safety_checker = dummy_checker
    
    with autocast("cuda"):
        spec = pipe(prompt, negative_prompt='', height=512, width=512).images[0]
    
    wav = wav_bytes_from_spectrogram_image(spec)
    
    buffered = BytesIO()
    spec.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    audio_buffer = BytesIO(wav)
    audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')

    return {'image_base64': image_base64, 'audio_base64': audio_base64}
