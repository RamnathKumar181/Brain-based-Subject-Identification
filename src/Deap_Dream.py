import os
from PIL import Image
import scipy.ndimage as nd
import numpy as np
import tqdm
import torch
from torch.optim import SGD
from torchvision import models
from utils import to_device
from torch.autograd import Variable
from misc_functions import preprocess_image, recreate_image, save_image


def dream(image, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
    image = Variable(torch.from_numpy(image), requires_grad=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = to_device(image,device)
    for i in range(iterations):
        model.zero_grad()
        out = model(image_tensor)
        loss = out.norm()
        # print(f"loss: {loss}| Iteration {i+1}")
        loss.backward()
        avg_grad = np.abs(image.grad.data.cpu().numpy()).mean()
        norm_lr = lr / avg_grad
        image.data += norm_lr * image.grad.data
        # image.data = clip(image.data)
        image.grad.data.zero_()
    return image.cpu().data.numpy()

def deep_dream(image, model, iterations, lr, octave_scale, num_octaves):
    """ Main deep dream method """
    image = image.unsqueeze(0).cpu().data.numpy()[0]
    # Extract image representations for each octave
    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            # Upsample detail to new octave dimension
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        # Add deep dream detail from previous octave to new base
        input_image = octave_base + detail
        # Get new deep dream image
        dreamed_image = dream(input_image, model, iterations, lr)
        # Extract deep dream details
        detail = dreamed_image - octave_base

    return dreamed_image
