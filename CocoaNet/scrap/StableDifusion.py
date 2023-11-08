#%%
import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda:0"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir="/local/scratch/jrs596/HF_DATASETS_CACHE")
pipe = pipe.to(device)


print(pipe)
#%%
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]  
    
    
image.save("/local/scratch/jrs596/StableDiffusion_images/astronaut_rides_horse.png")
# image.show()
# %%

# %%
for attr in dir(pipe):
    if "__" not in attr:  # Filter out built-in attributes
        print(attr)

# %%
unet_model = pipe.unet

print(unet_model)
# %%

import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, cache_dir="/local/scratch/jrs596/HF_DATASETS_CACHE")
pipe = pipe.to("cuda:0")
unet_model = pipe.unet

#%%
from diffusers import UNet2DModel
from torch.nn import functional as F
import torch

unet = UNet2DModel(
    sample_size=(48, 48),
    in_channels=84,
    out_channels=84).half()


noisy_encoded = torch.ones(4, 84, 45, 45).to(dtype=torch.float16).to("cuda:0")
noisy_encoded = F.pad(noisy_encoded, (1, 2, 1, 2))

encoded = torch.ones(4, 84, 45, 45).to(dtype=torch.float16).to("cuda:0")
encoded = F.pad(encoded, (1, 2, 1, 2))

unet = unet.to("cuda:0")
out = unet(sample=noisy_encoded, timestep=1)

criterion = torch.nn.MSELoss()

loss = criterion(out.sample, encoded)

print(loss)
# %%
