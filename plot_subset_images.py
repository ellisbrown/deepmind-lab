import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

np.random.seed(123)
random_indices = np.random.choice(np.arange(0, 300), size=20, replace=False)
random_indices = np.sort(random_indices)
root = "/home/alvin/research/github/16824/project/representation-exploration/experiment_data/"
folders = ["random_explore_negcontrast_task", "random_explore_poscontrast_task",
           "negcontrast_explore_negcontrast_task", "poscontrast_explore_poscontrast_task"]
for folder in folders:
    im_names = os.listdir(os.path.join(root, folder, "extracted_images"))

    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(10, 5))
    for i, ax in enumerate(axes.flatten()):
        im_name = im_names[random_indices[i]]
        im = Image.open(os.path.join(root, "extracted_images", im_name))
        im = np.array(im)
        ax.imshow(im[67:, :, :])  # remove "title" of saved matplotlib images
        ax.set_title("t={}".format(random_indices[i]))
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top = 0.5, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    # plt.show()
    plt.savefig(os.path.join(root, folder, "subset_images.png"))
