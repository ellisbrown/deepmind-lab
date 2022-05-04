
# @handle_legacy_interface(
#     weights=("pretrained", FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1),
#     weights_backbone=("pretrained_backbone", ResNet50_Weights.IMAGENET1K_V1),
# )

# https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py

import torchvision
from torchvision.models import resnet18, resnet50
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation.fcn import FCN, FCNHead
from torchvision.models._utils import IntermediateLayerGetter
import PIL
import numpy as np
import ipdb
import torch

def _fcn_resnet(
    backbone,
    num_classes: int,
    aux,
    # aux: Optional[bool],
) :
# -> FCN:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    # aux_classifier = FCNHead(1024, num_classes) if aux else None
    aux_classifier =  None
    classifier = FCNHead(2048, num_classes)
    return FCN(backbone, classifier, aux_classifier)


def fcn_resnet18(
    # *,
    # weights: Optional[FCN_ResNet50_Weights] = None,
    # progress: bool = True,
    # num_classes: Optional[int] = None,
    # aux_loss: Optional[bool] = None,
    # weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    # **kwargs: Any,
) :
# -> FCN:
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.
    Args:
        weights (FCN_ResNet50_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (ResNet50_Weights, optional): The pretrained weights for the backbone
    """
    # weights = FCN_ResNet50_Weights.verify(weights)
    # weights_backbone = ResNet50_Weights.verify(weights_backbone)

    # if weights is not None:
    #     weights_backbone = None
    #     num_classes = _ovewrite_value_param(num_classes, len(weights.meta["categories"]))
    #     aux_loss = _ovewrite_value_param(aux_loss, True)
    # elif num_classes is None:
    #     num_classes = 21

    num_classes = 21
    aux_loss = None

    # backbone = _resnet()
    backbone = resnet50(pretrained=False, replace_stride_with_dilation=[False, True, True])
    model = _fcn_resnet(backbone, num_classes, aux_loss)

    # if weights is not None:
    #     model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

# model = fcn_resnet18(num_classes = 3)
model = fcn_resnet50(num_classes = 3)

image = PIL.Image.open("sample_rgbd_images/0.png")
image = torchvision.transforms.ToTensor()(image).unsqueeze(0)

segmentation = np.load("sample_rgbd_images/0.npy")
segmentation = torch.tensor(segmentation).permute((2,0,1)).unsqueeze(0).float()

# ipdb.set_trace()
# model(image)
loss = torch.nn.CrossEntropyLoss()(model(image)["out"], segmentation)
print(loss)
