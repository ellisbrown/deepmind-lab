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

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from detectron2.engine.defaults import DefaultPredictor
from detectron2.config import get_cfg

import deepmind_lab

from utils import *


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
    def __init__(self, device, lr=0.001, batch_size=8, buffer_size=90):
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = []
        self.buffer_idx = 0
        self.num_updates = 0
        self.warmup_steps = 2

        # NOTE: Tune ALL parameters of network, DON'T FREEZE BACKBONE
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 4)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

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

    def calc_contrastive_values(self, cur_image, other_images=None):
        if other_images is None:
            if len(self.buffer) == 0:
                raise BufferEmptyError("Buffer is empty!")

            # NOTE: Beware of self.buffer_size, and whether you can fit this all into GPU memory
            other_images = torch.stack(self.buffer)
        cur_feat = self.model(cur_image.unsqueeze(0))
        other_feats = self.model(other_images)
        normalized_dot_prod = torch.sum(cur_feat * other_feats, dim=1) / (torch.norm(cur_feat, dim=1) * torch.norm(other_feats, dim=1))
        return normalized_dot_prod

    def calc_curiosity(self, images):
        try:
            curiosity = torch.stack([
                -1 * torch.mean(self.calc_contrastive_values(images[i]))
                for i in range(len(images))
            ]).detach().cpu().numpy().flatten()
            # print(np.array2string(curiosity, precision=2))
        except BufferEmptyError:
            curiosity = np.zeros(len(images))

        return curiosity

    def update(self, inp, ignore=None):
        """
        Perform contrastive update comparing input image with buffer. Add the input
        image to the buffer afterwards.

        """
        # always add to buffer
        import ipdb
        ipdb.set_trace()  # TODO: are these batches? if so, need to separate
        self.add_to_buffer(inp)

        # TODO: if poor results, only contrast with images collected K steps ago
        if len(self.buffer) < self.batch_size:
            # not enough data in buffer to do a batch update
            pass

        else:
            self.num_updates += 1
            buffer_entries = torch.stack(random.sample(self.buffer, self.batch_size))  # default replace=False
            normalized_dot_prod = self.calc_contrastive_values(inp, buffer_entries)
            loss = torch.mean(normalized_dot_prod)
            print("Contrastive loss:", loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


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
    def __init__(self, device, lr=0.001, batch_size=16, buffer_size=90):
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.im_buffer = []
        self.mask_buffer = []
        self.buffer_idx = 0
        self.num_updates = 0
        self.warmup_steps = 5
        # Segmentation model predicts logits
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

        # NOTE: Tune ALL parameters of network, DON'T FREEZE BACKBONE
        self.model = models.segmentation.fcn_resnet50(num_classes=1)  # binary mask
        model_path = "fcn_resnet50_coco_pretrained.pth"
        model_weights = torch.load(model_path)
        del model_weights["classifier.4.weight"], model_weights["classifier.4.bias"]
        self.model.load_state_dict(model_weights, strict=False)
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

            loss = self.bce_loss(self.calc_mask(batch_images.to(DEVICE)).cpu(), batch_masks.to(torch.float32)).mean()
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
        curiosity = self.bce_loss(self.calc_mask(images), masks.to(torch.float32))
        return curiosity.mean(dim=(1, 2)).detach().cpu().numpy()


class Explorer(object):
    def __init__(self, device, save_freq, vision_task: SegmentationVisionTask):
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

    def get_image_and_mask(self, env, transform=True):
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

    def get_surrounding_images(self, env):
        all_images = []
        all_masks = []
        im, mask = self.get_image_and_mask(env)
        all_images.append(im)
        all_masks.append(mask)

        interesting_images = []
        # To verify no double-counting, pass transform=True into self.get_obs() and view each image
        # 1st action already taken above (straight forward)
        # that leaves action_space - 1 actions left to take
        for i in range(len(self.action_space) - 1):
            env.step(LOOKAROUND_ACTION)
            im, mask = self.get_image_and_mask(env)
            all_images.append(im)
            all_masks.append(mask)

        # need 1 more action to return to initial view
        # verified by viewing current state after this action and comparing
        # to images[0]
        env.step(LOOKAROUND_ACTION)
        return all_images, all_masks

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
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        np.save(f"{self.save_dir}/all_images.npy", self.all_images)
        np.save(f"{self.save_dir}/all_action_distribs1.npy", self.all_action_distribs_rnd)
        np.save(f"{self.save_dir}/all_action_distribs2.npy", self.all_action_distribs_vision)
        np.save(f"{self.save_dir}/all_intrinsic_reward1.npy", self.all_reward_rnd)
        np.save(f"{self.save_dir}/all_intrinsic_reward2.npy", self.all_reward_contrast)
        np.save(f"{self.save_dir}/all_actions.npy", self.all_actions)
        np.save(f"{self.save_dir}/all_states.npy", self.all_states)
        np.save(f"{self.save_dir}/interesting_images.npy", self.interesting_images)
        print("Step %d Saved to %s" % (rollout_step, self.save_dir))

        # Save pretrained vision model, don't save fc layer so when loading this model, must set fc to None
        vision_model_state_dict = self.vision_task.model.state_dict()
        vision_model_state_dict.pop('fc.weight', None)
        vision_model_state_dict.pop('fc.bias', None)
        torch.save(vision_model_state_dict, f"{self.save_dir}/pretrained_vision_model_step_{rollout_step}.pt")


class RNDExplorer(Explorer):
    def __init__(self, use_ensemble, rollout_horizon, top_k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_ensemble = use_ensemble
        self.rollout_horizon = rollout_horizon
        self.top_k = top_k  # top_k / rollout_horizon = fraction of images to use for vision task
        self.lr = 1e-3
        self.n_train = 10
        self.eps1 = 0.0  # p(RND greedy)
        self.eps2 = 1.0  # p(contrastive vision task greedy)
        self.step = 0
        self.num_updates = 0
        self.rnd_warmup_steps = 3
        self.debug_save_freq = 3

        self.interesting_masks_gt = []
        self.interesting_masks_pred = []

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

    def update(self, batch):
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
        return init_loss, final_loss

    def get_action_distrib(self, env):
        """
        Always get action distrib, whether we actually use this to pick
        action or not (ie: if warm start enough) is left for the caller

        """
        env.step(NOOP_ACTION, num_steps=3)
        images, masks = self.get_surrounding_images(env)
        input_images = torch.stack(images).to(self.device)
        input_masks = torch.stack(masks).to(self.device)
        # print(env.observations()["DEBUG.POS.TRANS"])
        # print(env.observations()["DEBUG.POS.ROT"])

        # RND distribution
        rnd_rewards = torch.square(self.predictor(input_images) -
                                        self.target(input_images)).sum(dim=-1)
        rnd_rewards = rnd_rewards.detach().cpu().numpy().flatten()
        rnd_action_distrib = np.exp(rnd_rewards) / np.sum(np.exp(rnd_rewards))

        if isinstance(self.vision_task, SegmentationVisionTask):
            vision_rewards = self.vision_task.calc_curiosity(input_images, input_masks)
        elif isinstance(self.vision_task, ContrastiveVisionTask):
            vision_rewards = self.vision_task.calc_curiosity(input_images)
        else:
            raise ValueError("Unknown vision task")
        vision_action_distrib = np.exp(vision_rewards) / np.sum(np.exp(vision_rewards))

        return rnd_rewards, rnd_action_distrib, vision_rewards, vision_action_distrib, images, masks

    def rollout(self, env):
        # TODO: should replay buffer always use the forward-looking image, or
        #   rather pick all/random view(s) at each state?
        chosen_views = [self.get_obs(env)]
        all_views = []
        all_masks = []
        all_rnd_reward = np.array([])
        all_vision_reward = np.array([])

        # Take actions while collecting images for RND training
        for k in range(self.rollout_horizon):
            self.step += 1
            # cur_state = self.obs_to_im(self.get_obs(env))
            # plt.imshow(cur_state)
            # plt.show()

            with torch.no_grad():
                # batch x output_feat -> batch x 1
                rnd_rewards, rnd_action_distrib, vision_rewards, vision_action_distrib, images, masks = (
                    self.get_action_distrib(env))

            # TODO: epsilon greedy? or just sample from action distrib?
            rand_p = np.random.uniform()
            is_random = False
            if rand_p < self.eps1 and self.num_updates >= self.rnd_warmup_steps:
                direction_action = np.argmax(rnd_action_distrib)
            elif rand_p < self.eps1 + self.eps2 and self.vision_task.num_updates >= self.vision_task.warmup_steps:
                direction_action = np.argmax(vision_action_distrib)
            else:
                direction_action = np.random.randint(0, len(rnd_action_distrib))
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

            self.take_action(env, direction_action)
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

            chosen_views.append(self.get_obs(env))  # used to train RND
            all_views += images
            all_masks += masks
            all_rnd_reward = np.append(all_rnd_reward, rnd_rewards)
            all_vision_reward = np.append(all_vision_reward, vision_rewards)

            # Store agent state and action taken
            if self.num_updates % self.debug_save_freq == 0:
                orig_images = [self.obs_to_im(obs) for obs in images]
                self.all_images.append(orig_images)
                self.all_action_distribs_rnd.append(rnd_action_distrib)
                self.all_action_distribs_vision.append(vision_action_distrib)
                self.all_reward_rnd.append(rnd_rewards)
                self.all_reward_contrast.append(vision_rewards)
                self.all_actions.append(direction_action)
                self.all_states.append(self.get_state(env))

            # try:
            #     plt.imshow(im)
            #     plt.draw()
            #     plt.pause(0.2)
            # except KeyboardInterrupt:
            #     exit()

        return chosen_views, all_views, all_masks, all_rnd_reward, all_vision_reward

    def explore(self, env, n_steps):
        # wait for initial "halo" around agent to disappear
        env.step(NOOP_ACTION, num_steps=200)


        rollout_step = 0
        step = 0
        pbar = tqdm(total=n_steps)

        while step < n_steps:
            # NOTE: Reset Replay Buffer to only contain recently visited states
            chosen_views, chosen_masks, all_views, all_masks, all_rnd_reward, all_vision_reward = self.rollout(env)
            step += self.rollout_horizon
            rollout_step += 1
            pbar.update(self.rollout_horizon)

            if rollout_step % self.save_freq == 0:
                self.save_results(rollout_step)

            # Update Predictor Network n_train times using the same replay buffer
            if not np.isclose(self.eps1, 0, atol=1e-6):
                init_loss, final_loss = self.update(torch.stack(chosen_views))
                print("(%d) RND %.3f -> %.3f" % (step + 1, init_loss, final_loss))

            # Use top k images to train vision task
            # if self.num_updates >= self.rnd_warmup_steps:  # NOTE: removed this since can be used vision task to explore using eps2
            # top_image_idxs = np.argsort(all_rnd_reward)[-self.top_k:]
            if isinstance(self.vision_task, ContrastiveVisionTask):
                top_image_idxs = np.arange(len(chosen_views))
                for im_idx in top_image_idxs:
                    # Training only on "interesting" images
                    self.vision_task.update(all_views[im_idx], all_masks[im_idx])
                    self.interesting_images.append(self.obs_to_im(all_views[im_idx]))

                    # plt.imshow(self.interesting_images[-1])
                    # plt.title("Interesting image %d" % len(self.interesting_images))
                    # plt.draw()
                    # plt.pause(0.2)

            elif isinstance(self.vision_task, SegmentationVisionTask):
                # batch_images = torch.stack([all_views[im_idx] for im_idx in top_image_idxs])
                # batch_masks_gt = torch.stack([all_masks[im_idx] for im_idx in top_image_idxs])
                chosen_views = torch.stack(chosen_views)
                chosen_masks = torch.stack(chosen_masks)

                del all_views, all_masks, all_rnd_reward, all_vision_reward
                torch.cuda.empty_cache()
                gc.collect()
                self.vision_task.update(chosen_views, chosen_masks)
                del chosen_views, chosen_masks
                torch.cuda.empty_cache()
                gc.collect()
                #
                # if self.vision_task.num_updates % self.debug_save_freq == 0:
                #     with torch.no_grad():
                #         batch_masks_pred = torch.sigmoid(self.vision_task.calc_mask(batch_images))
                #     for i in range(batch_masks_pred.shape[0]):
                #         self.interesting_images.append(self.obs_to_im(batch_images[i]))
                #         self.interesting_masks_gt.append(batch_masks_gt[i].cpu().numpy())
                #         self.interesting_masks_pred.append(batch_masks_pred[i].cpu().numpy())

        self.save_results(rollout_step)

    def save_results(self, rollout_step):
        super().save_results(rollout_step)
        np.save(f"{self.save_dir}/interesting_masks_gt.npy", self.interesting_masks_gt)
        np.save(f"{self.save_dir}/interesting_masks_pred.npy", self.interesting_masks_pred)

class RNDExplorerPolicy(Explorer):
    PolicyActionSpace = {
        'look_left': _action(-142, 0, 0, 0, 0, 0, 0),  # 15 degrees
        'look_right': _action(142, 0, 0, 0, 0, 0, 0),
        'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
        'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
        'forward': _action(0, 0, 0, 1, 0, 0, 0),
        'backward': _action(0, 0, 0, -1, 0, 0, 0),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Define RND policy


class RandomExplorer(Explorer):
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


class ExplorerV2(Explorer):
    def take_action(self, env, ang_deg):
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


class RNDExplorerV2(RNDExplorer, ExplorerV2):
    def __init__(self, region_method="rpn", *args, **kwargs):
        # ExplorerV2.__init__(self, *args, **kwargs)  # redundant, but necessary if ExplorerV2 init changes
        RNDExplorer.__init__(self, *args, **kwargs)
        self.debug_save_freq = 1  # only saving one view, so can afford to save more often
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

        self.im_resize = transforms.Resize((256, 256))
        self.apply_transforms = transforms.Compose([
            transforms.ToPILImage(),
            self.im_resize,
            transforms.ToTensor(),
            transforms.Normalize(im_mean, im_std),
        ])

    def bbox_to_angle(self, bbox):
        # horizontal FOV is 90 degrees (https://github.com/deepmind/lab/issues/232)
        # since vertical is 90 and aspect_ratio = 512/512=1
        # also verified in collect_images.py
        left, top, right, bottom = bbox
        cx = (left + right) / 2
        return (cx - IM_SIZE/2) / (IM_SIZE/2) * 90  # if cx == W/2, ang = 0, if cx == 0, ang = -90

    def get_action_distrib(self, env):
        """
        Always get action distrib, whether we actually use this to pick
        action or not (ie: if warm start enough) is left for the caller

        """
        env.step(NOOP_ACTION, num_steps=3)
        img_raw, mask = self.get_image_and_mask(env, transform=False)
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
        all_rnd_curiosity = []
        all_vision_curiosity = []
        for i, bbox in enumerate(bboxes):
            left, top, right, bottom = bbox
            img_crop = img_raw[top:bottom, left:right]
            mask_crop = mask[top:bottom, left:right]
            img_crops.append(img_crop)
            mask_crops.append(mask_crop.cpu().numpy())

            input_img_crop = self.apply_transforms(img_crop).unsqueeze(0).to(self.device)
            input_mask_crop = self.im_resize(mask_crop.unsqueeze(0)).to(self.device)  # H x W -> 1 x H x W

            # RND distribution
            rnd_curiosity = torch.square(self.predictor(input_img_crop) -
                                            self.target(input_img_crop)).sum(dim=-1)

            if isinstance(self.vision_task, SegmentationVisionTask):
                vision_curiosity = self.vision_task.calc_curiosity(input_img_crop, input_mask_crop)
            elif isinstance(self.vision_task, ContrastiveVisionTask):
                vision_curiosity = self.vision_task.calc_curiosity(input_img_crop)
            else:
                raise ValueError("Unknown vision task")

            all_rnd_curiosity.append(rnd_curiosity.item())
            all_vision_curiosity.append(vision_curiosity.item())

        # Get action distrib
        all_rnd_curiosity = np.array(all_rnd_curiosity)
        all_vision_curiosity = np.array(all_vision_curiosity)
        rnd_action_distrib = np.exp(all_rnd_curiosity) / np.sum(np.exp(all_rnd_curiosity))
        vision_action_distrib = np.exp(all_vision_curiosity) / np.sum(np.exp(all_vision_curiosity))
        return all_rnd_curiosity, rnd_action_distrib, all_vision_curiosity, vision_action_distrib, self.apply_transforms(img_raw).to(self.device), mask, bboxes

    def rollout(self, env):
        # TODO: should replay buffer always use the forward-looking image, or
        #   rather pick all/random view(s) at each state?
        chosen_views = []
        chosen_masks = []
        all_bboxes = []
        all_rnd_reward = np.array([])
        all_vision_reward = np.array([])

        # Take actions while collecting images for RND training
        for k in range(self.rollout_horizon):
            self.step += 1
            # cur_state = self.obs_to_im(self.get_obs(env))
            # plt.imshow(cur_state)
            # plt.show()

            with torch.no_grad():
                # batch x output_feat -> batch x 1
                rnd_rewards, rnd_action_distrib, vision_rewards, vision_action_distrib, cur_img, cur_mask, bboxes = (
                    self.get_action_distrib(env))

            chosen_views.append(cur_img.cpu())
            chosen_masks.append(cur_mask.cpu())
            all_bboxes.append(bboxes)

            # TODO: epsilon greedy? or just sample from action distrib?
            rand_p = np.random.uniform()
            is_random = False
            if rand_p < self.eps1 and self.num_updates >= self.rnd_warmup_steps:
                region_idx = np.argmax(rnd_action_distrib)
            elif rand_p < self.eps1 + self.eps2 and self.vision_task.num_updates >= self.vision_task.warmup_steps:
                region_idx = np.argmax(vision_action_distrib)
            else:
                region_idx = np.random.randint(0, len(rnd_action_distrib))
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
            self.take_action(env, direction_angle)
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

            all_rnd_reward = np.append(all_rnd_reward, rnd_rewards)
            all_vision_reward = np.append(all_vision_reward, vision_rewards)

            # Store agent state and action taken
            if self.num_updates % self.debug_save_freq == 0:
                orig_images = self.obs_to_im(cur_img)
                self.all_images.append(orig_images)
                self.all_action_distribs_rnd.append(rnd_action_distrib)
                self.all_action_distribs_vision.append(vision_action_distrib)
                self.all_reward_rnd.append(rnd_rewards)
                self.all_reward_contrast.append(vision_rewards)
                self.all_actions.append(region_idx)  # Save index of bbox selected
                self.all_bboxes.append(bboxes)
                self.all_states.append(self.get_state(env))
                self.interesting_masks_gt.append(cur_mask.cpu().numpy())
                with torch.no_grad():
                    pred_mask = torch.sigmoid(self.vision_task.calc_mask(cur_img.unsqueeze(0)))
                self.interesting_masks_pred.append(pred_mask[0].cpu().numpy())
            # try:
            #     plt.imshow(im)
            #     plt.draw()
            #     plt.pause(0.2)
            # except KeyboardInterrupt:
            #     exit()

        all_views = None
        all_masks = None
        return chosen_views, chosen_masks, all_views, all_masks, all_rnd_reward, all_vision_reward

    def save_results(self, rollout_step):
        RNDExplorer.save_results(self, rollout_step)
        # Also save bounding boxes
        np.save(f"{self.save_dir}/all_bboxes.npy", self.all_bboxes)


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
    # vision_task = ContrastiveVisionTask(device=DEVICE, lr=3e-5)

    save_freq = 10
    rollout_horizon = 4
    top_k = 4  # top_k / rollout_horizon = fraction of images to use for vision task
    batch_size = 8
    vision_task = SegmentationVisionTask(device=DEVICE, lr=1e-3, batch_size=batch_size)

    use_ensemble = False
    # explorer = RNDExplorer(device=DEVICE, use_ensemble=use_ensemble, vision_task=vision_task,
    #                        rollout_horizon=rollout_horizon, top_k=top_k, save_freq=save_freq)
    explorer = RNDExplorerV2(device=DEVICE, use_ensemble=use_ensemble, vision_task=vision_task,
                           rollout_horizon=rollout_horizon, top_k=top_k, save_freq=save_freq)
    # explorer = RandomExplorer(device=DEVICE, vision_task=vision_task, rollout_horizon=rollout_horizon,
    #                           top_k=top_k, save_freq=save_freq)

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
