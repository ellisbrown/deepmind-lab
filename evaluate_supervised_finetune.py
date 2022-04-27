import torch
import torchvision
import torchvision.models as models
import torch.nn.functional as F

import numpy as np
import ipdb
import os
from PIL import Image

def finetune_model(model, train_dataset):

    n_out = len(train_dataset.classes)
    # TODO initialize this layer
    model.fc = torch.nn.Linear(512, n_out)

    # copied these args from hw2/task2.py
    train_loader = torch.utils.data.DataLoader(        
        train_dataset,
        batch_size=1,   # batchsize is one for this implementation
        shuffle=True,
    )
        # num_workers=4,
        # pin_memory=True,
        # sampler=None,
        # drop_last=True)
    
    # only optimize the final fc
    optimizer = torch.optim.SGD([t for m,t in model.named_parameters() if m[:3] == "fc."], 0.01)

    nepochs = 2
    for _ in range(nepochs):
        for _, data in enumerate(train_loader):
            image, label = data
            model_out = model(image)

            loss = F.cross_entropy(model_out, label)
            print("loss =", loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model

def eval_model(model, eval_dataset):
    model.eval()

    eval_loader = torch.utils.data.DataLoader(        
        eval_dataset,
        batch_size=1,
        shuffle=True,
    )

    accuracies = []
    for _, data in enumerate(eval_loader):
        image, label = data
        model_out = model(image)

        accuracy = label.argmax(dim = -1) == model_out.argmax(dim = -1)
        accuracies += accuracy.float().tolist()
    
    return np.mean(accuracies)

###

# Load both models, finetune both models, evaluate both model accuracies
def main(saved_model_path = "experiment_data/final_random/pretrained_vision_model_step_180.pt", 
            images_foldername = "evaluation_images"):

    # Load standard model
    pretrained_standard_model = models.resnet18(pretrained=True)

    # Load our model
    our_model = models.resnet18(pretrained=False)
    our_model_dict = torch.load(saved_model_path, map_location=torch.device('cpu'))
    missing, unexpected = our_model.load_state_dict(our_model_dict, strict=False)
    # some asserts for strict=False bookkeeping
    assert len(missing) == 2
    assert len(unexpected) == 0
    for m in missing:
        assert m[:3] == "fc."

    # Finetune both models
    train_dataset = torchvision.datasets.ImageFolder(images_foldername, transform = torchvision.transforms.ToTensor())
    ft_our_model = finetune_model(our_model, train_dataset)
    ft_standard_model = finetune_model(pretrained_standard_model, train_dataset)

    # Evaluate both model
    eval_dataset = torchvision.datasets.ImageFolder(images_foldername, transform = torchvision.transforms.ToTensor())
    our_result = eval_model(ft_our_model, eval_dataset)
    print("our result accuracy is", our_result)
    pretrained_standard_result = eval_model(ft_standard_model, eval_dataset)
    print("pretrained standard result accuracy is", pretrained_standard_result)

if __name__ == "__main__":
    main()





"""
David's old garbage code
"""

# class FineTunedModel(nn.Module):
#     def __init__(self, model, model_out_size):
#         self.pretrained = model
#         self.finetuned = torch.nn.Linear(model_out_size, model_out_size)
#         # initialize self.finetuned
    
#     def foward(self, x):
#         x = self.pretrained(x)
#         x = self.finetuned(x)
#         return x

# class ClassifierDataset(torch.utils.data.Dataset):
#     def __init__(self, images_dir = "boring_images"):
#         # https://stackoverflow.com/questions/2632205/how-to-count-the-number-of-files-in-a-directory-using-python

#         self.image_fnames = [os.path.join(images_dir, name) for name in os.listdir(images_dir)] # if os.path.isfile(name)]
#         self.num_images = len(self.image_fnames)
#         self.images_dir = images_dir

#     def __len__(self):
#         return self.num_images

#     def __getitem__(self, idx):
#         image = Image.open(self.image_fnames[idx])
#         image = torchvision.transforms.ToTensor()(image)

#         label = 
#         return {"image": image,
#                 "label": label}
