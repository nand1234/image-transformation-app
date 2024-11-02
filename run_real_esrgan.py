import sys
from unittest.mock import MagicMock

# Patch torchvision.transforms.functional_tensor
sys.modules['torchvision.transforms.functional_tensor'] = MagicMock()
sys.modules['torchvision.transforms.functional_tensor'].rgb_to_grayscale = \
    lambda x: x.mean(dim=-3, keepdim=True)

# Now import the rest of the modules
import argparse
import torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import os
import cv2
import urllib.request
from tqdm import tqdm

# Define the custom torch.load to force weights_only=True
_original_torch_load = torch.load

real_esrgan_model_name = 'RealESRGAN_x4plus.pth'

def download_weights(url, filename):
    weights_dir = 'weights'
    weights_path = os.path.join(weights_dir, filename)
    
    if not os.path.exists(weights_path):
        print(f"Downloading {filename}...")
        os.makedirs(weights_dir, exist_ok=True)
        
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, weights_path, reporthook=lambda b, bsize, tsize: t.update(bsize))
        
        print(f"{filename} downloaded successfully.")
    
    return weights_path

def get_model():
    # Fixed model architecture for x4 model
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    return model


def custom_torch_load(model_path, map_location=None, weights_only=True, *args, **kwargs):
    return _original_torch_load(model_path, map_location=map_location, weights_only=weights_only, *args, **kwargs)

def run_real_esrgan(img_array, scale=4):
    torch.load = custom_torch_load
    download_Realesrgan_weights()  # Ensure this function downloads and saves the weights correctly
    model_path = f"weights/{real_esrgan_model_name}"  # Ensure this path is correct

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=get_model(),
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True if torch.cuda.is_available() else False
    )

    # Run enhancement
    output, _ = upsampler.enhance(img_array, outscale=scale)

    # Optionally, revert torch.load to its original form after the function executes
    torch.load = _original_torch_load

    return output


def download_Realesrgan_weights():
    weights_url = f"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/{real_esrgan_model_name}"
    return download_weights(weights_url, real_esrgan_model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('output', type=str, default='results', help='Output folder')
    parser.add_argument('scale', type=int, default=4, help='Upscaling factor (2/3/4)')
    args = parser.parse_args()

    img_array = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img_array is None:
        raise ValueError(f"Failed to read image: {args.input}")
    
    output = run_real_esrgan(img_array, args.scale)
    cv2.imwrite(args.output, output)

    print(f"Image processed successfully. Output saved to {args.output}")
