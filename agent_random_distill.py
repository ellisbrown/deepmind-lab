import argparse
import random
import numpy as np
import six
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
from datetime import datetime
import ipdb
import gc
import cv2
import time
import typing as t
import json

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg

import deepmind_lab

from utils import *
from self_sup_pretraining import contrastive_loss
from simsiam.transform import TwoCropsTransform, GaussianBlur

def _action(*entries):
    return np.array(entries, dtype=np.intc)

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

IM_SIZE = 256  # height and width of the image
MAX_ABS_STEPSIZE = 512
FIXED_CAMERA_STEPSIZE = 487  # takes approx 7 steps to rotate 360 degrees
STEPS_TO_FULLY_ROTATE = 7
# FIXED_CAMERA_STEPSIZE = 142
# STEPS_TO_FULLY_ROTATE = 24
STEPS_PER_DEG = (FIXED_CAMERA_STEPSIZE * STEPS_TO_FULLY_ROTATE) / 360.0
LOOKAROUND_ACTION = np.array([FIXED_CAMERA_STEPSIZE, 0, 0, 0, 0, 0, 0], dtype=np.intc)
FORWARD_ACTION = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.intc)
BACKWARD_ACTION = np.array([0, 0, 0, -1, 0, 0, 0], dtype=np.intc)
LOOKDOWN_ACTION = np.array([0, 20, 0, 0, 0, 0, 0], dtype=np.intc)
NOOP_ACTION = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.intc)
OBSERVATION_MODE = "DEBUG.CAMERA_INTERLEAVED.PLAYER_VIEW_NO_RETICLE"

FORWARD_ANG_THRESH = 10  # move forward if angle < 10 deg

cuda = torch.cuda.is_available()
DEVICE = "cuda" if cuda else "cpu"
if cuda:
    print("CUDA GPU!")
else:
    print("CPU!")

ActionSpace = {
    'look_left': _action(-20, 0, 0, 0, 0, 0, 0),
    'look_right': _action(20, 0, 0, 0, 0, 0, 0),
    'look_up': _action(0, 10, 0, 0, 0, 0, 0),
    'look_down': _action(0, -10, 0, 0, 0, 0, 0),
    'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
    'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
    'forward': _action(0, 0, 0, 1, 0, 0, 0),
    'backward': _action(0, 0, 0, -1, 0, 0, 0),
}

im_mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE)
im_std = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE)
apply_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IM_SIZE, IM_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(im_mean, im_std),
        ])
save_im_resize = transforms.Resize(100)


class BufferEmptyError(Exception):
    pass


class ContrastiveVisionTask(object):
    # image contrastive: mask size
    # store feature representations
    # 1. compare with entire buffer treat the entire buffer as negatives
    # 2. randomly sample pairs of images from entire buffer
    # objects are always spinning, so taking two images, we can use the motion
    # to get a segmentation mask and use Deepak's "Learning Features by Watching Objects Move"
    # Early on RND gives bad images
    # You sent
    # Try low lr initial and increase throughout exploration
    # Try using 1/4 jmagenet data during exploration fine tuning to not completely forget imagenet data
    def __init__(self, device, neg_contrast, loss_type, lr=0.001, batch_size=8, buffer_size=90):
        self.device = device
        self.neg_contrast = neg_contrast
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = []
        self.buffer_idx = 0
        self.num_updates = 0
        self.warmup_steps = 2

        self.loss_type = loss_type
        if loss_type == "cosine":
            # assume x and y are sufficiently different images, minimize cosine similarity between them
            # this means curiosity = -1*cosine similarity
            self.loss_fn = lambda x, y: torch.mean(torch.sum(x * y, dim=1) / (torch.norm(x, dim=1) * torch.norm(y, dim=1)))
        elif loss_type == "mse":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unknown loss type: {}".format(loss_type))

        # define random augmentations to produce two different images for positive contrast
        if not neg_contrast:
            self.positive_aug = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=im_mean, std=im_std)
            ])

        # NOTE: Tune ALL parameters of network, DON'T FREEZE BACKBONE
        self.model = models.resnet50(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 4)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.7, patience=2, verbose=True)

    def add_to_buffer(self, entry):
        """
        Add an entry to the buffer.
        :param entry:
            MoCo: image feature
            Random sampling: image
        """
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(entry)
        else:
            self.buffer[self.buffer_idx % self.buffer_size] = entry
        self.buffer_idx += 1

    def calc_contrastive_loss(self, cur_image, other_images=None):
        if other_images is None:
            if len(self.buffer) == 0:
                raise BufferEmptyError("Buffer is empty!")

            other_images = torch.stack(random.sample(self.buffer, min(self.batch_size, len(self.buffer))))
        cur_feat = self.model(cur_image.unsqueeze(0).to(self.device))
        other_feats = self.model(other_images.to(self.device))
        return self.loss_fn(cur_feat, other_feats)

    def calc_curiosity(self, images):
        try:
            curiosity = torch.stack([
                self.calc_contrastive_loss(images[i])
                for i in range(len(images))
            ]).detach().cpu().numpy().flatten()
            if self.loss_type == "cosine":
                curiosity = -1 * curiosity
            # print(np.array2string(curiosity, precision=2))
        except BufferEmptyError:
            curiosity = np.zeros(len(images))

        return curiosity

    def update(self, images):
        """
        Perform contrastive update comparing input image with buffer. Add the input
        image to the buffer afterwards.

        """
        self.num_updates += 1
        for i in range(images.shape[0]):
            self.add_to_buffer(images[i].cpu())

        # Rather than train directly on the input samples, store into buffer
        # to reduce influence of correlated samples
        # "The Challenges of Continuous Self-Supervised Learning"
        if len(self.buffer) < self.batch_size:
            # not enough data in buffer to do a batch update
            pass

        else:
            self.num_updates += 1

            batch_images = torch.stack(random.sample(self.buffer, self.batch_size))
            if self.neg_contrast:
                batch_features = self.model(batch_images.to(self.device))
                loss = contrastive_loss(batch_features, num_sampled=self.batch_size // 2)
                del batch_features
                # loss = self.calc_contrastive_loss()
            else:
                batch_features1 = self.model(
                    torch.stack([self.positive_aug(im) for im in batch_images]).to(self.device))
                batch_features2 = self.model(
                    torch.stack([self.positive_aug(im) for im in batch_images]).to(self.device))
                loss = self.loss_fn(batch_features1, batch_features2)
                del batch_features1, batch_features2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(loss.item())
            print("Contrastive loss:", loss.item())

            del loss, batch_images
            torch.cuda.empty_cache()
            gc.collect()


class SegmentationVisionTask(object):
    # image contrastive: mask size
    # store feature representations
    # 1. compare with entire buffer treat the entire buffer as negatives
    # 2. randomly sample pairs of images from entire buffer
    # objects are always spinning, so taking two images, we can use the motion
    # to get a segmentation mask and use Deepak's "Learning Features by Watching Objects Move"
    # Early on RND gives bad images
    # You sent
    # Try low lr initial and increase throughout exploration
    # Try using 1/4 jmagenet data during exploration fine tuning to not completely forget imagenet data
    def __init__(self, device, is_depth, lr=0.001, batch_size=16, buffer_size=90):
        self.is_depth = is_depth
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.im_buffer = []
        self.mask_buffer = []
        self.buffer_idx = 0
        self.num_updates = 0
        self.warmup_steps = 5

        if is_depth:
            self.loss_fn = torch.nn.MSELoss(reduction='none')
        else:
            # Segmentation model predicts logits
            self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

        # NOTE: Tune ALL parameters of network, DON'T FREEZE BACKBONE
        self.model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=1)  # binary mask
        # model_path = "fcn_resnet50_coco_pretrained.pth"
        # model_weights = torch.load(model_path)
        # del model_weights["classifier.4.weight"], model_weights["classifier.4.bias"]
        # self.model.load_state_dict(model_weights, strict=False)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.7, patience=2, verbose=True)

    def add_to_buffer(self, image, mask):
        """
        Add an entry to the buffer.
        :param entry:
            MoCo: image feature
            Random sampling: image
        """
        if len(self.im_buffer) < self.buffer_size:
            self.im_buffer.append(image)
            self.mask_buffer.append(mask)
        else:
            self.im_buffer[self.buffer_idx % self.buffer_size] = image
            self.mask_buffer[self.buffer_idx % self.buffer_size] = mask

        self.buffer_idx += 1

    def calc_mask(self, images):
        # image: B x C x H x W
        # output: B x H x W
        return self.model(images)["out"].squeeze(1)

    def update(self, images, masks):
        """
        Perform supervised update comparing input image with segmentation mask. 
        TODO?: Add the input image to the buffer afterwards.
        """
        self.num_updates += 1
        for i in range(images.shape[0]):
            self.add_to_buffer(images[i].cpu(), masks[i].cpu())

        # Rather than train directly on the input samples, store into buffer
        # to reduce influence of correlated samples
        # "The Challenges of Continuous Self-Supervised Learning"
        if len(self.im_buffer) < self.batch_size:
            # not enough data in buffer to do a batch update
            pass

        else:
            self.num_updates += 1

            batch_images = torch.stack(random.sample(self.im_buffer, self.batch_size))
            batch_masks = torch.stack(random.sample(self.mask_buffer, self.batch_size))

            loss = self.loss_fn(self.calc_mask(batch_images.to(DEVICE)).cpu(), batch_masks.to(torch.float32)).mean()
            print("Seg loss:", loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(loss.item())

            del loss, batch_images, batch_masks
            torch.cuda.empty_cache()
            gc.collect()

    def calc_curiosity(self, images, masks):
        # image: B x C x H x W
        # segmentation: B x H x W
        # output: B
        curiosity = self.loss_fn(self.calc_mask(images), masks.to(torch.float32))
        return curiosity.mean(dim=(1, 2)).detach().cpu().numpy()


class Explorer(object):
    def __init__(self, device, save_freq, vision_task: t.Union[ContrastiveVisionTask, SegmentationVisionTask]):
        """
        :param device: torch.device
        :param vision_task: vision task to be trained on "interesting" images
            through the exploration process
        """
        super().__init__()
        self.device = device
        self.vision_task = vision_task
        self.save_freq = save_freq  # save every N rollout_step's NOTE: (not steps)
        now = datetime.now()
        now_str = now.strftime("%Y_%m_%d_%H:%M:%S")
        self.save_dir = "experiment_data/%s" % now_str
        os.mkdir(self.save_dir)

        # [0(initial view), 1, ... STEPS_TO_FULLY_ROTATE-1(final view before returning to inital view)]
        # takes STEPS_TO_FULLY_ROTATE to full rotate, so to avoid double-counting
        # initial rotation,
        self.action_space = np.arange(STEPS_TO_FULLY_ROTATE)
        self.num_actions = len(self.action_space)
        # takes 15 sets of 10 forward steps to reach from one end to the other of vlr_env2, so move 1/5 of the way
        self.num_steps_forward = 3 * 10  # num steps to move forward in a chosen direction

        self.all_images = []
        self.all_action_distribs_rnd = []
        self.all_action_distribs_vision = []
        self.all_reward_rnd = []
        self.all_reward_contrast = []
        self.all_actions = []
        self.all_states = []
        self.interesting_images = []

    def get_obs(self, env, transform=True):
        if transform:
            return apply_transforms(env.observations()[OBSERVATION_MODE]).to(self.device)
        else:
            return env.observations()[OBSERVATION_MODE]

    def get_state(self, env):
        return np.concatenate([env.observations()["DEBUG.POS.TRANS"], env.observations()["DEBUG.POS.ROT"]])

    @staticmethod
    def obs_to_im(obs):
        obs = obs * im_std.view(3, 1, 1).to(obs.device) + im_mean.view(3, 1, 1).to(obs.device)
        # if resize:
        #     obs = save_im_resize(obs)
        return obs.permute(1, 2, 0).detach().cpu().numpy()

    def get_image(self, env, transform=True):
        env.step(NOOP_ACTION, num_steps=1)  # THIS IS NECESSARY! Env doesn't render fast enough?
        return self.get_obs(env, transform=transform)

    def get_surrounding_images(self, env):
        all_images = []
        im = self.get_image(env)
        all_images.append(im)

        interesting_images = []
        # To verify no double-counting, pass transform=True into self.get_obs() and view each image
        # 1st action already taken above (straight forward)
        # that leaves action_space - 1 actions left to take
        for i in range(len(self.action_space) - 1):
            env.step(LOOKAROUND_ACTION)
            im = self.get_image(env)
            all_images.append(im)

        # need 1 more action to return to initial view
        # verified by viewing current state after this action and comparing
        # to images[0]
        env.step(LOOKAROUND_ACTION)
        return all_images

    def take_action(self, env, direction_action: int):
        # Re-orient in desired direction
        # if direction_action==0, don't rotate, just move straight forward
        env.step(LOOKAROUND_ACTION, num_steps=direction_action)

        # print("Before stepping forward")
        # cur_state = self.obs_to_im(self.get_obs(env))
        # plt.imshow(cur_state)
        # plt.show()

        # Take N steps forward only if already facing desired direction
        if direction_action == 0:
            env.step(FORWARD_ACTION, num_steps=self.num_steps_forward)
            # wait for some timesteps to stop moving
            # https://github.com/deepmind/lab/issues/236#issuecomment-1119820048
            env.step(NOOP_ACTION, num_steps=20)
            # image, mask = self.get_image_and_mask(env)
            # show_image(self.obs_to_im(image))
            # show_image(mask)

        # print("final state after taking action:")
        # cur_state = self.obs_to_im(self.get_obs(env))
        # plt.imshow(cur_state)
        # plt.show()

    def get_action_distrib(self, env):
        raise NotImplementedError("Base class")

    def explore(self, env, n_steps):
        raise NotImplementedError("Base class")

    def save_results(self, rollout_step):
        np.save(f"{self.save_dir}/all_images.npy", self.all_images)
        np.save(f"{self.save_dir}/all_actions.npy", self.all_actions)
        np.save(f"{self.save_dir}/all_states.npy", self.all_states)
        np.save(f"{self.save_dir}/interesting_images.npy", self.interesting_images)
        print("Step %d Saved to %s" % (rollout_step, self.save_dir))

        # Save pretrained vision model, don't save fc layer so when loading this model, must set fc to None
        vision_model_state_dict = self.vision_task.model.state_dict()
        vision_model_state_dict.pop('fc.weight', None)
        vision_model_state_dict.pop('fc.bias', None)
        torch.save(vision_model_state_dict, f"{self.save_dir}/pretrained_vision_model_step_{rollout_step}.pt")


class ExplorerV2(Explorer):
    def __init__(self, region_method="rpn", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_bboxes = []
        if region_method == "random":
            self.get_regions = lambda img: get_random_regions(img)
        elif region_method == "grid":
            self.get_regions = None
            pass  # don't use
        elif region_method == "rpn":
            cfg = get_cfg()    # obtain detectron2's default config
            cfg.merge_from_file("rpn_config.yaml")
            self.rpn = DefaultPredictor(cfg)  # Region Proposal Network
            self.get_regions = lambda img: get_rpn_regions(self.rpn, img)

    def take_action(self, env, ang_deg, spin_around=False):
        if spin_around:
            for _ in range(STEPS_TO_FULLY_ROTATE // 2):
                env.step(LOOKAROUND_ACTION)
            return

        # Re-orient in desired direction
        # TODO: visually verify that we perform a rotation with desired angle
        signed_steps = int(STEPS_PER_DEG * ang_deg)
        while signed_steps != 0:
            action_step = np.clip(signed_steps, -MAX_ABS_STEPSIZE, MAX_ABS_STEPSIZE)
            env.step(np.array([action_step, 0, 0, 0, 0, 0, 0], dtype=np.intc))
            signed_steps -= action_step  # action_step < 0  =>  -action_steps > 0

        # Take N steps forward only if already facing desired direction
        if abs(ang_deg) < FORWARD_ANG_THRESH:
            env.step(FORWARD_ACTION, num_steps=self.num_steps_forward)
            env.step(NOOP_ACTION, num_steps=20)  # wait for velocity to settle to 0

    def bbox_to_angle(self, bbox):
        # horizontal FOV is 90 degrees (https://github.com/deepmind/lab/issues/232)
        # since vertical is 90 and aspect_ratio = 512/512=1
        # also verified in collect_images.py
        left, top, right, bottom = bbox
        cx = (left + right) / 2
        return (cx - IM_SIZE/2) / (IM_SIZE/2) * 90  # if cx == W/2, ang = 0, if cx == 0, ang = -90

    def save_results(self, rollout_step):
        super().save_results(rollout_step)
        np.save(f"{self.save_dir}/all_bboxes.npy", self.all_bboxes)

class RandomExplorer(ExplorerV2):
    def __init__(self, rollout_horizon, top_k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Random Explorer doesn't actually have any "rollout", but this is to ensure
        # that Random Explorer saves images at the same rate as RND so we can
        # fairly compare the two
        self.rollout_horizon = rollout_horizon
        self.top_k = top_k

    def get_action_distrib(self, env):
        # random actions with equal prob
        action_distrib = np.ones(len(self.action_space)) / len(self.action_space)
        return action_distrib

    def rollout(self, env):
        replay_buffer = [self.get_obs(env)]
        for _ in range(self.rollout_horizon):
            # random action
            direction_action = np.random.choice(self.action_space)
            self.take_action(env, direction_action)
            replay_buffer.append(self.get_obs(env))

            # Store agent's state and taken action
            images = self.get_surrounding_images(env)
            orig_images = [self.obs_to_im(obs) for obs in images]
            self.all_images.append(orig_images)
            self.all_states.append(self.get_state(env))
            self.all_actions.append(direction_action)

            if not env.is_running():
                # Env stopped??? Why?
                ipdb.set_trace()
                break

        return replay_buffer

    def explore(self, env, n_steps):
        # spin around to get rid of initial "halo" around agent
        self.get_surrounding_images(env)
        self.get_surrounding_images(env)
        self.get_surrounding_images(env)

        rollout_step = 0
        step = 0
        pbar = tqdm(total=n_steps)

        # Store initial state
        images = self.get_surrounding_images(env)
        orig_images = [self.obs_to_im(obs) for obs in images]
        self.all_images.append(orig_images)
        self.all_states.append(self.get_state(env))

        while step < n_steps:
            # NOTE: Reset Replay Buffer to only contain recently visited states
            replay_buffer = self.rollout(env)
            step += len(replay_buffer)
            rollout_step += 1
            pbar.update(len(replay_buffer))

            if rollout_step % self.save_freq == 0:
                self.save_results(rollout_step)

            # Draw top_k images randomly to train vision task
            image_idxs = np.random.choice(len(replay_buffer), self.top_k)
            for im_idx in image_idxs:
                self.vision_task.update(replay_buffer[im_idx])
                self.interesting_images.append(self.obs_to_im(replay_buffer[im_idx]))  # for visualization

        self.save_results(rollout_step)


class ExplorerContrastive(ExplorerV2):
    def __init__(self, use_rnd, use_ensemble, rollout_horizon, top_k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_rnd = use_rnd
        self.use_ensemble = use_ensemble
        self.rollout_horizon = rollout_horizon
        self.top_k = top_k  # top_k / rollout_horizon = fraction of images to use for vision task
        self.lr = 1e-3
        self.n_train = 5
        self.eps1 = 0.0  # p(RND greedy)
        self.eps2 = 1.0  # p(contrastive vision task greedy)
        self.step = 0
        self.num_updates = 0
        self.rnd_warmup_steps = 3
        self.debug_save_freq = 3

        # only add "interesting" images to vision task after at least 1 update of RND networks
        self.min_steps_intrinsic_reward = self.rollout_horizon

        # TODO: Use ensemble? predictor would minimize average error across ensemble
        #   but seems possible certain states would have a lower bound on error
        #   no matter how many times they are trained on
        self.target = models.resnet18(pretrained=True).to(self.device)
        self.predictor = models.resnet18(pretrained=True).to(self.device)
        for param in self.predictor.parameters():
            param.requires_grad = False
        # NOTE: define model.fc AFTER so only fc has requires_grad=True
        if self.use_ensemble:
            pass
        else:
            self.target.fc = nn.Linear(512, 128).to(self.device)
            self.predictor.fc = nn.Linear(512, 128).to(self.device)
        params_to_update = []
        for p in self.predictor.parameters():
            if p.requires_grad == True:  # only fc params will have requires_grad=True
                params_to_update.append(p)

        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(params_to_update, lr=self.lr)

    def update_rnd(self, batch):
        init_loss = None
        self.num_updates += 1
        for i in range(self.n_train):
            loss = torch.nn.functional.mse_loss(self.predictor(batch), self.target(batch))

            # calc gradients, update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if init_loss is None:
                init_loss = loss.item()

        final_loss = loss.item()
        print("RND %.3f -> %.3f" % (init_loss, final_loss))
        return init_loss, final_loss

    def get_action_distrib(self, env):
        """
        Always get action distrib, whether we actually use this to pick
        action or not (ie: if warm start enough) is left for the caller

        """
        env.step(NOOP_ACTION, num_steps=3)
        img_raw = self.get_image(env, transform=False)
        img_transformed = apply_transforms(img_raw).to(self.device)
        H, W, _ = img_raw.shape
        bboxes = self.get_regions(img_raw)

        # """
        # NOTE: I think the below is wrong...
        # Depth not necessary to extract angle.
        # Since pixels lie on image plane (ie: homogeneous coordinates),
        # their coordinate is [u, v, 1] rather than [x, y, z].
        # Can calculate theta = atan2(1, u) to get horizontal angle.
        # """
        # depth_img = env.observations()["RGBD_INTERLEAVED"]

        # Calculate RND error for each box
        img_crops = []
        all_curiosity = []
        for i, bbox in enumerate(bboxes):
            left, top, right, bottom = bbox
            img_crop = img_raw[top:bottom, left:right]
            img_crops.append(img_crop)

            input_img_crop = apply_transforms(img_crop).unsqueeze(0).to(self.device)

            if self.use_rnd:
                curiosity = torch.square(self.predictor(input_img_crop) -
                                         self.target(input_img_crop)).sum(dim=-1)

            else:
                curiosity = self.vision_task.calc_curiosity(input_img_crop)

            all_curiosity.append(curiosity.item())

        # Get action distrib
        all_curiosity = np.array(all_curiosity)
        action_distrib = np.exp(all_curiosity) / np.sum(np.exp(all_curiosity))
        return all_curiosity, action_distrib, img_transformed, bboxes, None

    def rollout(self, env):
        # TODO: should replay buffer always use the forward-looking image, or
        #   rather pick all/random view(s) at each state?
        chosen_views = []
        all_aux_info = []
        all_bboxes = []
        all_reward = np.array([])

        # Take actions while collecting images for RND training
        for k in range(self.rollout_horizon):
            self.step += 1
            # cur_state = self.obs_to_im(self.get_obs(env))
            # plt.imshow(cur_state)
            # plt.show()

            with torch.no_grad():
                # batch x output_feat -> batch x 1
                rewards, action_distrib, cur_img, bboxes, aux_info = self.get_action_distrib(env)

            chosen_views.append(cur_img.cpu())
            all_aux_info.append(aux_info)
            all_bboxes.append(bboxes)

            # TODO: epsilon greedy? or just sample from action distrib?
            rand_p = np.random.uniform()
            is_random = False
            spin_around = False   # don't do this, causes agent to keep looking back at cow
            # if rand_p < self.eps1 and self.num_updates >= self.rnd_warmup_steps:
            #     region_idx = np.argmax(rnd_action_distrib)
            if rand_p < self.eps1 + self.eps2 and self.vision_task.num_updates >= self.vision_task.warmup_steps:
                region_idx = np.argmax(action_distrib)

                # Nothing interesting to see, spin around
                # print(np.max(vision_action_distrib) - np.min(vision_action_distrib))
                # if np.max(vision_action_distrib) - np.min(vision_action_distrib) < 0.004:
                #     spin_around = True

            else:
                region_idx = np.random.randint(0, len(action_distrib))
                is_random = True
            # direction_action = np.random.choice(self.action_space, p=action_distrib)

            # cur_state_again = self.obs_to_im(self.get_obs(env))
            # plt.imshow(cur_state_again)
            # plt.show()

            # print("desired:")
            # assert len(self.action_space) == len(images)
            # desired_im = self.obs_to_im(images[direction_action])
            # plt.imshow(desired_im)
            # plt.show()
            if not is_random and self.vision_task.num_updates >= 10:
                show = False
                # ipdb.set_trace()

                if show:
                    images, masks = self.get_surrounding_images(env)
                    input_images = torch.stack(images).to(self.device)
                    input_masks = torch.stack(masks).to(self.device)
                    vals = self.vision_task.calc_curiosity(input_images, input_masks)
                    for i in range(len(vals)):
                        show_image(self.obs_to_im(images[i]))
                        show_image(masks[i], title="val: {}".format(vals[i]))
                # for i in range(len(vals)):
                #     plt.imshow(self.obs_to_im(self.vision_task.buffer[i]))
                #     plt.title("reward: {:.2f}".format(-1 * vals[i]))
                #     plt.show()

            direction_angle = self.bbox_to_angle(bboxes[region_idx])
            action = (region_idx, spin_around)
            self.take_action(env, direction_angle, spin_around=spin_around)
            # print("Actions: ", self.action_space)
            # print("Action distrib: ", np.array2string(action_distrib, precision=2))

            # plt.imshow(self.obs_to_im(self.get_obs(env)))
            # plt.title("Action index: {}, is_random: {}".format(direction_action, is_random))
            # plt.draw()
            # plt.pause(0.2)

            if not env.is_running():
                # Env stopped??? Why?
                ipdb.set_trace()
                break

            all_reward = np.append(all_reward, rewards)

            # Store agent state and action taken
            if self.num_updates % self.debug_save_freq == 0:
                self.log_info(env, rewards, action_distrib, cur_img, bboxes, action, aux_info)
            # try:
            #     plt.imshow(im)
            #     plt.draw()
            #     plt.pause(0.2)
            # except KeyboardInterrupt:
            #     exit()

        rollout_info = dict(images=chosen_views, all_aux_info=all_aux_info)
        return rollout_info

    def log_info(self, env, rewards, action_distrib, cur_img, bboxes, action, aux_info):
        orig_images = self.obs_to_im(cur_img)
        self.all_images.append(orig_images)
        if self.use_rnd:
            self.all_action_distribs_rnd.append(action_distrib)
            self.all_reward_rnd.append(rewards)
        else:
            self.all_action_distribs_vision.append(action_distrib)
            self.all_reward_contrast.append(rewards)

        self.all_actions.append(action)  # Save index of bbox selected
        self.all_bboxes.append(bboxes)
        self.all_states.append(self.get_state(env))

    def explore(self, env, n_steps):
        # wait for initial "halo" around agent to disappear
        env.step(NOOP_ACTION, num_steps=200)
        rollout_step = 0
        step = 0
        pbar = tqdm(total=n_steps)

        while step < n_steps:
            # NOTE: Reset Replay Buffer to only contain recently visited states
            rollout_info = self.rollout(env)
            step += self.rollout_horizon
            rollout_step += 1
            pbar.update(self.rollout_horizon)

            if rollout_step % self.save_freq == 0:
                self.save_results(rollout_step)

            if self.use_rnd:
                self.update_rnd(torch.stack(rollout_info["images"]).to(self.device))

            self.update_vision_task(rollout_info)

        self.save_results(rollout_step)

    def update_vision_task(self, rollout_info):
        self.vision_task.update(torch.stack(rollout_info["images"]))
        torch.cuda.empty_cache()
        gc.collect()

    def save_results(self, rollout_step):
        super().save_results(rollout_step)
        np.save(f"{self.save_dir}/all_action_distribs_rnd.npy", self.all_action_distribs_rnd)
        np.save(f"{self.save_dir}/all_action_distribs_vision.npy", self.all_action_distribs_vision)
        np.save(f"{self.save_dir}/all_reward_rnd.npy", self.all_reward_rnd)
        np.save(f"{self.save_dir}/all_reward_contrast.npy", self.all_reward_contrast)


class ExplorerMasks(ExplorerContrastive):
    def __init__(self, is_depth, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_save_freq = 1  # only saving one view, so can afford to save more often

        self.resize = transforms.Resize((IM_SIZE, IM_SIZE))
        self.is_depth = is_depth
        if is_depth:
            self.get_image_and_mask = self.get_image_and_depth
        else:
            self.get_image_and_mask = self.get_image_and_motion_mask

        self.interesting_masks_gt = []
        self.interesting_masks_pred = []

    def get_image_and_motion_mask(self, env, transform=True):
        env.step(NOOP_ACTION, num_steps=1)  # THIS IS NECESSARY! Env doesn't render fast enough?
        im1 = self.get_obs(env, transform=False)
        env.step(NOOP_ACTION, num_steps=3)  # wait for N timesteps to let objects rotate
        im2 = self.get_obs(env, transform=False)
        thresh, diff_img = calc_SSIM_mask(im1, im2)
        # masks = calc_nls_mask(im1, im2)
        # show_image(im1)
        # show_image(im2)
        # show_image(masks[1])
        # show_image(thresh)
        # show_image(diff_img)
        if transform:
            im2 = apply_transforms(im2).to(self.device)

        return im2, torch.from_numpy(thresh).to(self.device)  # H x W

    def get_image_and_depth(self, env, transform=True):
        env.step(NOOP_ACTION, num_steps=1)  # THIS IS NECESSARY! Env doesn't render fast enough?
        # im = self.get_obs(env, transform=transform)
        assert IM_SIZE == 256  # TODO: make this work for other sizes
        im = env.observations()[OBSERVATION_MODE][0:220, :, :]
        im = self.resize(torch.from_numpy(im).permute(2, 0, 1))
        if transform:
            im = apply_transforms(im).to(self.device)
        else:
            im = im.permute(1, 2, 0).numpy()
        depth = torch.from_numpy(env.observations()["RGBD_INTERLEAVED"][0:220, :, -1])
        depth = self.resize(depth.unsqueeze(0).float()).to(self.device)[0] / 255.0

        return im, depth  # H x W

    def get_action_distrib(self, env):
        """
        Always get action distrib, whether we actually use this to pick
        action or not (ie: if warm start enough) is left for the caller

        """
        env.step(NOOP_ACTION, num_steps=3)
        img_raw, mask = self.get_image_and_mask(env, transform=False)
        img_transformed = apply_transforms(img_raw).to(self.device)
        H, W, _ = img_raw.shape
        bboxes = self.get_regions(img_raw)

        # """
        # NOTE: I think the below is wrong...
        # Depth not necessary to extract angle.
        # Since pixels lie on image plane (ie: homogeneous coordinates),
        # their coordinate is [u, v, 1] rather than [x, y, z].
        # Can calculate theta = atan2(1, u) to get horizontal angle.
        # """
        # depth_img = env.observations()["RGBD_INTERLEAVED"]

        # Calculate RND error for each box
        img_crops = []
        mask_crops = []
        all_curiosity = []
        for i, bbox in enumerate(bboxes):
            left, top, right, bottom = bbox
            img_crop = img_raw[top:bottom, left:right]
            mask_crop = mask[top:bottom, left:right]
            img_crops.append(img_crop)
            mask_crops.append(mask_crop.cpu().numpy())

            input_img_crop = apply_transforms(img_crop).unsqueeze(0).to(self.device)
            input_mask_crop = self.resize(mask_crop.unsqueeze(0)).to(self.device)  # H x W -> 1 x H x W

            if self.use_rnd:
                curiosity = torch.square(self.predictor(input_img_crop) -
                                         self.target(input_img_crop)).sum(dim=-1)

            else:
                curiosity = self.vision_task.calc_curiosity(input_img_crop, input_mask_crop)

            all_curiosity.append(curiosity.item())

        # Get action distrib
        all_curiosity = np.array(all_curiosity)
        action_distrib = np.exp(all_curiosity) / np.sum(np.exp(all_curiosity))
        return all_curiosity, action_distrib, img_transformed, bboxes, mask.cpu()

    def log_info(self, env, rewards, action_distrib, cur_img, bboxes, action, debug_info):
        super().log_info(env, rewards, action_distrib, cur_img, bboxes, action, debug_info)
        cur_mask = debug_info
        self.interesting_masks_gt.append(cur_mask.cpu().numpy())
        with torch.no_grad():
            pred_mask = torch.sigmoid(self.vision_task.calc_mask(cur_img.unsqueeze(0)))
        self.interesting_masks_pred.append(pred_mask[0].cpu().numpy())

    def update_vision_task(self, rollout_info):
        self.vision_task.update(torch.stack(rollout_info["images"]), torch.stack(rollout_info["all_aux_info"]))
        torch.cuda.empty_cache()
        gc.collect()

    def save_results(self, rollout_step):
        super().save_results(rollout_step)
        np.save(os.path.join(self.save_dir, "interesting_masks_gt.npy"), self.interesting_masks_gt)
        np.save(os.path.join(self.save_dir, "interesting_masks_pred.npy"), self.interesting_masks_pred)


def run(length, width, height, fps, level, record, demo, demofiles, video):
    """Spins up an environment and runs the random agent."""
    # Extremely hacky way to move custom object/level files to site-packages
    src_root = "/home/alvin/research/github/16824/project/representation-exploration/deepmind-lab"
    dst_root = "/home/alvin/catkin_ws/src/meta_cobot_learning/rnp/venv/lib/python3.6/site-packages/deepmind_lab/baselab"
    files = [
        f"game_scripts/levels/{level}.lua",
        "game_scripts/common/vlr_objects.lua"
    ]
    for fname in files:
        src = os.path.join(src_root, fname)
        dst = os.path.join(dst_root, fname)
        shutil.copyfile(src, dst)

    config = {
        'fps': str(fps),
        'width': str(width),
        'height': str(height)
    }
    if record:
        config['record'] = record
    if demo:
        config['demo'] = demo
    if demofiles:
        config['demofiles'] = demofiles
    if video:
        config['video'] = video

    # NOTE: RGBD_INTERLEAVED or RGBD also available, potentially train 3D visual features
    # other_states = ["DEBUG.POS.TRANS", "DEBUG.POS.ROT", "DEBUG.CAMERA_INTERLEAVED.PLAYER_VIEW",
    #                 "DEBUG.CAMERA_INTERLEAVED.PLAYER_VIEW_NO_RETICLE"]
    # "DEBUG.CAMERA_INTERLEAVED.TOP_DOWN": unfortunately not good to use
    env = deepmind_lab.Lab(level, [OBSERVATION_MODE, "DEBUG.POS.ROT", "DEBUG.POS.TRANS", "RGBD_INTERLEAVED"], config=config)
    env.reset()

    # Empirically approximate a good action value for seeing different views
    # count = 0
    # image = env.observations()[OBSERVATION_MODE]
    # plt.imshow(image)
    # plt.draw()
    # plt.pause(1.5)
    # for i in range(STEPS_TO_FULLY_ROTATE):
    #     try:
    #         env.step(LOOKAROUND_ACTION)
    #         image = env.observations()[OBSERVATION_MODE]
    #         count += 1
    #         plt.imshow(image)
    #         plt.draw()
    #         plt.pause(1.5)
    #     except KeyboardInterrupt:
    #         exit()

    # save_freq = 10
    # rollout_horizon = 4
    # top_k = 2  # top_k / rollout_horizon = fraction of images to use for vision task

    save_freq = 10
    rollout_horizon = 4
    top_k = 4  # top_k / rollout_horizon = fraction of images to use for vision task
    batch_size = 8
    use_ensemble = False
    use_rnd = False
    is_contrast = True
    is_depth = None
    neg_contrast = None
    contrast_loss_type = None

    if is_contrast:
        neg_contrast = False
        contrast_loss_type = "cosine"  # mse
        vision_task = ContrastiveVisionTask(device=DEVICE, lr=3e-5, batch_size=batch_size, neg_contrast=neg_contrast, loss_type=contrast_loss_type)
        explorer = ExplorerContrastive(device=DEVICE, use_ensemble=use_ensemble, vision_task=vision_task,
                                       rollout_horizon=rollout_horizon, top_k=top_k, save_freq=save_freq, use_rnd=use_rnd)
    else:
        is_depth = True
        vision_task = SegmentationVisionTask(device=DEVICE, lr=1e-3, batch_size=batch_size, is_depth=is_depth)
        explorer = ExplorerMasks(use_rnd=use_rnd, device=DEVICE, use_ensemble=use_ensemble, vision_task=vision_task,
                                 rollout_horizon=rollout_horizon, top_k=top_k, save_freq=save_freq, is_depth=is_depth)

    is_random = False
    # is_random = True
    # explorer = RandomExplorer(device=DEVICE, vision_task=vision_task, rollout_horizon=rollout_horizon,
    #                           top_k=top_k, save_freq=save_freq)
    args = {
        "save_freq": save_freq,
        "rollout_horizon": rollout_horizon,
        "top_k": top_k,
        "batch_size": batch_size,
        "is_depth": is_depth,
        "use_ensemble": use_ensemble,
        "use_rnd": use_rnd,
        "is_contrast": is_contrast,
        "neg_contrast": neg_contrast,
        "is_random": is_random,
        "contrast_loss_type": contrast_loss_type,
    }

    # save to json
    with open(os.path.join(explorer.save_dir, "args.json"), "w") as f:
        json.dump(args, f)

    print(json.dumps(args, indent=4))

    explorer.explore(env, n_steps=length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=int, default=1000,
                        help='Number of steps to run the agent')
    parser.add_argument('--fps', type=int, default=60,
                        help='Number of frames per second')
    parser.add_argument('--runfiles_path', type=str, default=None,
                        help='Set the runfiles path to find DeepMind Lab data')
    parser.add_argument('--level_script', type=str,
                        default='tests/empty_room_test',
                        help='The environment level script to load')
    parser.add_argument('--record', type=str, default=None,
                        help='Record the run to a demo file')
    parser.add_argument('--demo', type=str, default=None,
                        help='Play back a recorded demo file')
    parser.add_argument('--demofiles', type=str, default=None,
                        help='Directory for demo files')
    parser.add_argument('--video', type=str, default=None,
                        help='Record the demo run as a video')

    args = parser.parse_args()
    if args.runfiles_path:
        deepmind_lab.set_runfiles_path(args.runfiles_path)
    run(args.length, IM_SIZE, IM_SIZE, args.fps, args.level_script,
        args.record, args.demo, args.demofiles, args.video)
