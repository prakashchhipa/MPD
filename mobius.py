import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

import torch.nn.functional as F

import numpy as np
from PIL import Image

import torch, sys, json, random, os, pathlib

import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image


import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torch.nn.functional as F


class MobiusTransform_Improved(nn.Module):
    
    def __init__(self, a_real=1.0, a_imag=0.0, b_real=0.0, b_imag=0.0, 
                 d_real=1., d_imag=0., p=1., min_magnitude=0.2, max_magnitude=0.3, img_bck_ref = False,
                 only_top_down = False
                 ):
        print(f'Adding MobiusTransform Improved in the augmentation pipeline with probability {p} ')
        super(MobiusTransform_Improved, self).__init__()

        self.a_real = a_real
        self.a_imag = a_imag
        self.b_real = b_real
        self.b_imag = b_imag
        self.c_real = 0.0
        self.c_imag = 0.0
        self.d_real = d_real
        self.d_imag = d_imag
        self.p = p
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.img_bck_ref = img_bck_ref
        self.only_top_down = only_top_down
        
    def _randomize_c(self):
        """Randomly set values for c_real and c_imag based on the described conditions."""
        if self.only_top_down:
            component = random.choice(['c_imag', 'c_imag'])
        else:
            component = random.choice(['c_real', 'c_imag'])
        sign = random.choice([1, -1])
        magnitude = random.uniform(self.min_magnitude, self.max_magnitude)

        if component == 'c_real':
            c_real = magnitude * sign
            c_imag = 0.0
        else:
            c_imag = magnitude * sign
            c_real = 0.0

        return torch.tensor([c_real, c_imag])

    def likelihood(self) -> bool:
        if 0 <= self.p <= 1:
            return random.random() < self.p
        else:
            raise ValueError("Probability p should be in the range [0, 1]")
    
    
    
    
    def forward(self, image):
        with torch.no_grad():
            decision = self.likelihood()
            if decision:
                c_real, c_imag = self._randomize_c()
                width, height = image.size

                y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width))
                
                z_real = x
                z_imag = y

                # z_transformed = (a * z + b) / (c * z + d)
                num_real = self.a_real * z_real - self.a_imag * z_imag + self.b_real
                num_imag = self.a_real * z_imag + self.a_imag * z_real + self.b_imag

                den_real = c_real * z_real - c_imag * z_imag + self.d_real
                den_imag = c_real * z_imag + c_imag * z_real + self.d_imag

                z_transformed_real = (num_real * den_real + num_imag * den_imag) / (den_real**2 + den_imag**2)
                z_transformed_imag = (num_imag * den_real - num_real * den_imag) / (den_real**2 + den_imag**2)
                

                x_transformed = z_transformed_real
                y_transformed = z_transformed_imag

                grid = torch.stack((x_transformed, y_transformed), dim=-1).unsqueeze(0)
                
                np_image = np.array(image)
                if len(np_image.shape) == 2:
                    image_array = torch.tensor(np_image).unsqueeze(0).unsqueeze(1).float()
                else:
                    image_array = torch.tensor(np_image).unsqueeze(0).permute(0, 3, 1, 2).float()
                
                #interpolate with reference for black background
                if self.img_bck_ref:
                    image_transformed = F.grid_sample(image_array, grid, align_corners=True, padding_mode='reflection')
                else:
                    image_transformed = F.grid_sample(image_array, grid, align_corners=True)
                
                image_np = image_transformed[0].numpy()

                if image_np.shape[0] == 1:
                    image_np = np.squeeze(image_np, axis=0)
                else:
                    image_np = np.transpose(image_np, (1, 2, 0))

                    #print(f"Data type of image_np: {image_np.dtype}")
                    return Image.fromarray(np.uint8(image_np))
            else:
                return image