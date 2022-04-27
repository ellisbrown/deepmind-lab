import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
import numpy as np
import random
from torchvision import models
torch.set_default_dtype(torch.double)
import argparse
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

device = torch.device("cuda")
im_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
im_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
apply_transforms = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(im_mean, im_std),
])

class ImageNetPretrainedResNet(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        self.model = models.resnet18(pretrained=pretrained)
        self.num_classes = num_classes
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model = self.model.float()

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)


class RNDPretrainedResNet(nn.Module):
    def __init__(self, device, saved_model_path, num_classes=10):
        super().__init__()
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        self.model = models.resnet18(pretrained=False)
        in_features = self.model.fc.in_features

        self.model.fc = None
        our_model_dict = torch.load(saved_model_path, map_location=device)
        self.model.load_state_dict(our_model_dict, strict=False)
        
        self.num_classes = num_classes
        self.model.fc = torch.nn.Linear(in_features, self.num_classes)
        self.model = self.model.float()

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

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


def train(args, model, train_loader, test_loader):
    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.model.fc.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    cnt = 0
    for epoch in tqdm(range(args.epochs)):
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            
            # Set train mode
            model.train()

            # Get a batch of data
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate the loss
            loss = loss_fn(output, target)

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
        model.eval()
        accuracy = eval_dataset(model, args.device, test_loader)
        print(f"Test accuracy : {accuracy}")

        # accuracy = eval_dataset(model, args.device, train_loader)
        # print(f"Train accuracy : {accuracy}")
        model.train()

    final_accuracy = eval_dataset(model, args.device, test_loader)
    print(f"Final accuracy : {final_accuracy}")
    return final_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_images_foldername', type=str, default='./vlr_train_dataset_final',
                        help='train_images_foldername')
    parser.add_argument('--test_images_foldername', type=str, default='./vlr_test_dataset_final',
                        help='test_images_foldername')
    parser.add_argument('--saved_random_model_path', type=str, default='./experiment_data/final_random/pretrained_vision_model_step_198.pt',
                        help='saved_random_model_path')
    parser.add_argument('--saved_rnd_model_path', type=str, default='./experiment_data/final_RND/pretrained_vision_model_step_198.pt',
                        help='saved_rnd_model_path')
    parser.add_argument('--log_every', type=int, default= 25,
                        help='Log every')
    parser.add_argument('--val_every', type=int, default=1,
                        help='Val every')
    parser.add_argument('--epochs', type=int, default=2,
                        help='epochs')

    args = parser.parse_args()
    args.device = device

    # Init models
    pretrained_model_baseline_1 = ImageNetPretrainedResNet()
    pretrained_model_baseline_2 = ImageNetPretrainedResNet(pretrained=False)

    random_model_baseline = RNDPretrainedResNet(device=args.device, saved_model_path=args.saved_random_model_path)
    rnd_model = RNDPretrainedResNet(device=args.device, saved_model_path=args.saved_rnd_model_path)

    # Init dataset
    train_dataset = torchvision.datasets.ImageFolder(args.train_images_foldername, transform = apply_transforms)
    test_dataset = torchvision.datasets.ImageFolder(args.test_images_foldername, transform = apply_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    
    # Train and evaluate baseline model
    baseline_accuracy = train(args, pretrained_model_baseline_1, train_loader=train_loader, test_loader=test_loader)
    print(f"Pretrained True Baseline accuracy: {baseline_accuracy}")

    # Train and evaluate baseline model
    baseline_accuracy = train(args, pretrained_model_baseline_2, train_loader=train_loader, test_loader=test_loader)
    print(f"Pretrained False Baseline accuracy: {baseline_accuracy}")

    # Train and evaluate our model
    rnd_accuracy = train(args, rnd_model, train_loader=train_loader, test_loader=test_loader)
    print(f"RND accuracy: {rnd_accuracy}")

    # Train and evaluate baseline random model
    random_accuracy = train(args, random_model_baseline, train_loader=train_loader, test_loader=test_loader)
    print(f"Random model baseline accuracy: {random_accuracy}")
