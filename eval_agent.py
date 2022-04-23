import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms as transforms

from agent_random_distill import apply_transforms, RNDExplorer, DEVICE

def load_images(dir, fname=None, is_np=False):
    """Loads images from a directory."""
    images = []
    if is_np:
        all_images = np.load(os.path.join(dir, fname), allow_pickle=True)
        for i in range(all_images.shape[0]):
            images.append(all_images[i])
    else:
        for filename in os.listdir(dir):
            if filename.endswith(".png"):
                image = Image.open(os.path.join(dir, filename))
                images.append(np.array(image))
    return images

agent_images = load_images("experiment_data/2022_04_22_01:08:21", is_np=True, fname="all_images.npy")
actions = np.load("experiment_data/2022_04_22_01:08:21/all_actions.npy")
boring_images = load_images("boring_images")
boring_images = [RNDExplorer.obs_to_im(apply_transforms(torch.from_numpy(im).permute(2,0,1)).to(DEVICE))[np.newaxis] for im in boring_images]
boring_images = np.vstack(boring_images)

avg_error = 0
num = 0
for image, action in zip(agent_images, actions):
    error = np.mean(np.abs(image[action][np.newaxis] - boring_images))
    avg_error += error
    num += 1

print("Average error:", avg_error / num)

