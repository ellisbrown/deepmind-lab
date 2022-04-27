import numpy as np
import os
import shutil

dataset_dir = './vlr_dataset_final'
test_data_dir = './vlr_test_dataset_final'
train_data_dir = './vlr_train_dataset_final'

def make_safe_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


make_safe_dir(train_data_dir)
make_safe_dir(test_data_dir)

subfolders = os.listdir(dataset_dir)
for subfolder in subfolders:
    print(f"Processing {subfolder}")
    dataset_path = os.path.join(dataset_dir, subfolder)
    train_dataset_path = os.path.join(train_data_dir, subfolder)
    test_dataset_path = os.path.join(test_data_dir, subfolder)
    make_safe_dir(train_dataset_path)
    make_safe_dir(test_dataset_path)

    image_files = np.array(os.listdir(dataset_path))
    num_images = image_files.shape[0]
    indices = np.random.permutation(num_images)
    image_files = image_files[indices]

    num_to_split = int(np.floor(num_images * 0.7))

    # Copy train images
    train_split = image_files[:num_to_split]
    for train_image in train_split:
        src_path = os.path.join(dataset_path, train_image)
        dst_path = os.path.join(train_dataset_path, train_image)
        shutil.copyfile(src=src_path, dst=dst_path)  


    # Copy test images
    test_split = image_files[num_to_split:]
    for test_image in test_split:
        src_path = os.path.join(dataset_path, test_image)
        dst_path = os.path.join(test_dataset_path, test_image)
        shutil.copyfile(src=src_path, dst=dst_path) 
