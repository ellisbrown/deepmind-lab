import torch
import torchvision
import torchvision.models as models
# import torch.nn.functional as F

import numpy as np
# import ipdb
import os
# from PIL import Image

import argparse
import random
import numpy as np
# import six
# import matplotlib.pyplot as plt
import os
# import shutil
from tqdm import tqdm
from datetime import date
# import ipdb

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
# import utils

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DEVICE = torch.device("cuda")
im_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
im_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)
apply_transforms = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(im_mean, im_std),
])

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def contrastive_loss(features, num_sampled=8):
    # batch_size
    batch_size = features.shape[0]

    assert num_sampled < batch_size

    # Assuming features are shuffled
    inp_feat = features[:num_sampled]
    other_feat = features[num_sampled:]

    sum_feat = other_feat @ inp_feat.T
    norm_input = torch.norm(inp_feat, dim=1).reshape(1, -1).repeat_interleave(other_feat.shape[0], dim=0)
    norm_other = torch.norm(other_feat, dim=1).reshape(1, -1).repeat_interleave(inp_feat.shape[0], dim=0).T

    normalized_prod = sum_feat / (norm_input * norm_other)

    return torch.mean(normalized_prod)


@torch.no_grad()
def eval_dataset(model, device, test_loader):
    model.eval()
    success = 0.
    ncount = 0
    with torch.no_grad():
        for data, target in test_loader:
            ncount += data.shape[0]
            data, target = data.to(device), target.cpu()
            output = torch.argmax(torch.nn.functional.softmax(model(data)).cpu(), dim=1)
            del data
            success += torch.sum(output == target)
    accuracy = success / ncount
    model.train()
    return accuracy


def train_self_sup(args, model, loss_fn, optimizer, train_loader, test_loader):
    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)
    cnt = 0
    for epoch in tqdm(range(args.epochs)):
        for batch_idx, (data, _) in tqdm(enumerate(train_loader)):
            
            # Set train mode
            model.train()

            # Get a batch of data
            data = data.to(args.device)
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate the loss
            loss = loss_fn(output)

            # Calculate gradient w.r.t the loss
            loss.backward()

            # Optimizer takes one step
            optimizer.step()

            # # Log info
            if cnt % args.log_every == 0:
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))

            cnt += 1

        # Validation iteration
        # model.eval()
        # accuracy = eval_dataset(model, args.device, test_loader)
        # print(f"Test accuracy : {accuracy}")

        # # accuracy = eval_dataset(model, args.device, train_loader)
        # # print(f"Train accuracy : {accuracy}")
        # model.train()

    final_accuracy = eval_dataset(model, args.device, test_loader)
    print(f"Final accuracy : {final_accuracy}")
    return final_accuracy, model

def run_simple_contrastive(args, train_loader, test_loader):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 64)
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    _, model = train_self_sup(args, model, contrastive_loss, optimizer, train_loader, test_loader)
    save_model(args, model)


def save_model(args, model, tag="contrastive_loss"):
    # Save pretrained vision model
    vision_model_state_dict = model.state_dict()
    curr_time = date.today().strftime("%m_%d_%y_%H_%M_%S")
    torch.save(vision_model_state_dict, f"{args.save_model_path}/pretrained_oracle_vision_model_{tag}_epoch_{args.epochs}_batchsize_{args.batch_size}_time_{curr_time}.pt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all_images_foldername', type=str, default='./vlr_dataset_oracle_interesting_images',
                        help='Folder containing oracle images to train on')
    parser.add_argument('--save_model_path', type=str, default='./experiment_data/pretrained_oracle',
                        help='path to save the pretrained model')
    parser.add_argument('--self_supervised_method', type=str, default=None,
                            help="self supervised method to be used")
    parser.add_argument('--log_every', type=int, default= 25,
                        help='Log every')
    parser.add_argument('--val_every', type=int, default=1,
                        help='Val every')
    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--lr', type=int, default=1e-3,
                        help='learning rate')

    args = parser.parse_args()
    args.device = DEVICE

    # Init dataset
    train_dataset = torchvision.datasets.ImageFolder(args.all_images_foldername, transform = apply_transforms)
    
    # Both train and test are same
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Run Contrastive
    run_simple_contrastive(args, train_loader, test_loader)