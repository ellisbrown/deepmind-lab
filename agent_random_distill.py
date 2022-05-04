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

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

import deepmind_lab


def _action(*entries):
    return np.array(entries, dtype=np.intc)


seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

FIXED_CAMERA_STEPSIZE = 487  # takes approx 7 steps to rotate 360 degrees
STEPS_TO_FULLY_ROTATE = 7
LOOKAROUND_ACTION = np.array([FIXED_CAMERA_STEPSIZE, 0, 0, 0, 0, 0, 0], dtype=np.intc)
FORWARD_ACTION = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.intc)
BACKWARD_ACTION = np.array([0, 0, 0, -1, 0, 0, 0], dtype=np.intc)
LOOKDOWN_ACTION = np.array([0, 20, 0, 0, 0, 0, 0], dtype=np.intc)
NOOP_ACTION = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.intc)
OBSERVATION_MODE = "DEBUG.CAMERA_INTERLEAVED.PLAYER_VIEW_NO_RETICLE"

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
            transforms.CenterCrop(256),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(im_mean, im_std),
        ])
save_im_resize = transforms.Resize(100)


class BufferEmptyError(Exception):
    pass


class PretrainedVisionTask(object):
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
        self.buffer = []
        self.buffer_idx = 0
        self.num_updates = 0
        self.warmup_steps = 5

        # NOTE: Tune ALL parameters of network, DON'T FREEZE BACKBONE
        self.model = models.resnet18(pretrained=True)
        # TODO: LOWER THIS LATENT SPACE
            # guess 2-3 values for size of RND latent space (128, 64, 256)
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

    def update(self, inp):
        """
        Perform contrastive update comparing input image with buffer. Add the input
        image to the buffer afterwards.

        """
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

        # always add to buffer
        self.buffer.append(inp)

class PretrainedSegmentationVisionTask(object):
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
        self.buffer = []
        self.buffer_idx = 0
        self.num_updates = 0
        self.warmup_steps = 5

        # NOTE: Tune ALL parameters of network, DON'T FREEZE BACKBONE
        self.model = models.fcn_resnet50(pretrained=True)
        # TODO: LOWER THIS LATENT SPACE
            # guess 2-3 values for size of RND latent space (128, 64, 256)
        self.model = self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def update(self, image, segmentation):
        """
        Perform supervised update comparing input image with segmentation mask. 
        TODO?: Add the input image to the buffer afterwards.

        """

        self.num_updates += 1

        loss = torch.nn.CrossEntropyLoss()(self.model(image)["out"], segmentation)

        print("Supervised loss:", loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # TODO always add to buffer


class Explorer(object):
    def __init__(self, device, save_freq, vision_task: PretrainedVisionTask):
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
        self.num_steps_forward = 10  # num steps to move forward in a chosen direction

        self.all_images = []
        self.all_action_distribs_rnd = []
        self.all_action_distribs_contrast = []
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
        return save_im_resize(obs * im_std.view(3, 1, 1) + im_mean.view(3, 1, 1)
                                   ).permute(1, 2, 0).detach().cpu().numpy()

    def get_surrounding_images(self, env):
        images = [self.get_obs(env)]
        interesting_images = []
        # To verify no double-counting, pass transform=True into self.get_obs() and view each image
        # 1st action already taken above (straight forward)
        # that leaves action_space - 1 actions left to take
        for i in range(len(self.action_space) - 1):
            env.step(LOOKAROUND_ACTION)
            images.append(self.get_obs(env))

        # need 1 more action to return to initial view
        # verified by viewing current state after this action and comparing
        # to images[0]
        env.step(LOOKAROUND_ACTION)
        return images

    def take_action(self, env, direction_action: int):
        # Re-orient in desired direction
        # if direction_action==0, don't rotate, just move straight forward
        env.step(LOOKAROUND_ACTION, num_steps=direction_action)

        # print("Before stepping forward")
        # cur_state = self.obs_to_im(self.get_obs(env))
        # plt.imshow(cur_state)
        # plt.show()

        # Take N steps forward
        env.step(FORWARD_ACTION, num_steps=self.num_steps_forward)
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
        np.save(f"{self.save_dir}/all_action_distribs2.npy", self.all_action_distribs_contrast)
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
        self.rnd_warmup_steps = 5

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
        images = self.get_surrounding_images(env)
        # print(env.observations()["DEBUG.POS.TRANS"])
        # print(env.observations()["DEBUG.POS.ROT"])

        # RND distribution
        input_images = torch.stack(images).to(self.device)
        RND_intrinsic_reward = torch.square(self.predictor(input_images) -
                                        self.target(input_images)).sum(dim=-1)
        RND_intrinsic_reward = RND_intrinsic_reward.detach().cpu().numpy().flatten()
        RND_action_distrib = np.exp(RND_intrinsic_reward) / np.sum(np.exp(RND_intrinsic_reward))

        # Contrastive vision task distribution
        try:
            contrastive_intrinsic_reward = torch.stack([
                -1 * torch.mean(self.vision_task.calc_contrastive_values(images[i]))
                for i in range(len(images))
            ])
            contrastive_intrinsic_reward = contrastive_intrinsic_reward.detach().cpu().numpy().flatten()
            contrastive_action_distrib = np.exp(contrastive_intrinsic_reward) / np.sum(np.exp(contrastive_intrinsic_reward))
            print(np.array2string(contrastive_intrinsic_reward, precision=2))
        except BufferEmptyError:
            contrastive_intrinsic_reward = np.zeros(len(images))
            contrastive_action_distrib = np.ones(len(images)) / len(images)

        return RND_intrinsic_reward, RND_action_distrib, contrastive_intrinsic_reward, contrastive_action_distrib, images

    def rollout(self, env):
        # TODO: should replay buffer always use the forward-looking image, or
        #   rather pick all/random view(s) at each state?
        chosen_views = [self.get_obs(env)]
        all_views = []
        all_rnd_reward = np.array([])
        all_contrast_reward = np.array([])

        # Take actions while collecting images for RND training
        for k in range(self.rollout_horizon):
            self.step += 1
            # cur_state = self.obs_to_im(self.get_obs(env))
            # plt.imshow(cur_state)
            # plt.show()

            with torch.no_grad():
                # batch x output_feat -> batch x 1
                rnd_reward, rnd_action_distrib, contrast_reward, contrast_action_distrib, images = self.get_action_distrib(env)

            # TODO: epsilon greedy? or just sample from action distrib?
            rand_p = np.random.uniform()
            is_random = False
            if rand_p < self.eps1 and self.num_updates >= self.rnd_warmup_steps:
                direction_action = np.argmax(rnd_action_distrib)
            elif rand_p < self.eps1 + self.eps2 and self.vision_task.num_updates >= self.vision_task.warmup_steps:
                direction_action = np.argmax(contrast_action_distrib)
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
            if not is_random:
                ipdb.set_trace()
                plt.clf()
                plt.imshow(self.obs_to_im(self.get_obs(env)))
                plt.show()
                vals = self.vision_task.calc_contrastive_values(self.get_obs(env))
                for i in range(len(vals)):
                    plt.imshow(self.obs_to_im(self.vision_task.buffer[i]))
                    plt.title("reward: {:.2f}".format(-1 * vals[i]))
                    plt.show()

            self.take_action(env, direction_action)
            # print("Actions: ", self.action_space)
            # print("Action distrib: ", np.array2string(action_distrib, precision=2))

            plt.imshow(self.obs_to_im(self.get_obs(env)))
            plt.title("Action index: {}, is_random: {}".format(direction_action, is_random))
            plt.draw()
            plt.pause(0.2)

            if not env.is_running():
                # Env stopped??? Why?
                ipdb.set_trace()
                break

            chosen_views.append(self.get_obs(env))  # used to train RND
            all_views += images
            all_rnd_reward = np.append(all_rnd_reward, rnd_reward)
            all_contrast_reward = np.append(all_contrast_reward, contrast_reward)

            # Store agent state and action taken
            orig_images = [self.obs_to_im(obs) for obs in images]
            self.all_images.append(orig_images)
            self.all_action_distribs_rnd.append(rnd_action_distrib)
            self.all_action_distribs_contrast.append(contrast_action_distrib)
            self.all_reward_rnd.append(rnd_reward)
            self.all_reward_contrast.append(contrast_reward)
            self.all_actions.append(direction_action)
            self.all_states.append(self.get_state(env))

            # try:
            #     plt.imshow(im)
            #     plt.draw()
            #     plt.pause(0.2)
            # except KeyboardInterrupt:
            #     exit()

        return chosen_views, all_views, all_rnd_reward, all_contrast_reward

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
            chosen_views, all_views, all_rnd_reward, all_contrast_reward = self.rollout(env)
            step += self.rollout_horizon
            rollout_step += 1
            pbar.update(self.rollout_horizon)

            if rollout_step % self.save_freq == 0:
                self.save_results(rollout_step)

            # Update Predictor Network n_train times using the same replay buffer
            batch_states = torch.stack(chosen_views)
            init_loss, final_loss = self.update(batch_states)
            print("(%d) RND %.3f -> %.3f" % (step + 1, init_loss, final_loss))

            # Use top k images to train vision task
            if self.num_updates >= self.rnd_warmup_steps:
                top_image_idxs = np.argsort(all_rnd_reward)[-self.top_k:]
                for im_idx in top_image_idxs:
                    self.vision_task.update(all_views[im_idx])
                    self.interesting_images.append(self.obs_to_im(all_views[im_idx]))  # for visualization

                    plt.imshow(self.interesting_images[-1])
                    plt.title("Interesting image %d" % len(self.interesting_images))
                    plt.draw()
                    plt.pause(0.2)

        self.save_results(rollout_step)


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
    env = deepmind_lab.Lab(level, [OBSERVATION_MODE, "DEBUG.POS.ROT", "DEBUG.POS.TRANS"], config=config)
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

    save_freq = 10
    rollout_horizon = 16
    top_k = 4  # top_k / rollout_horizon = fraction of images to use for vision task
    use_ensemble = False
    vision_task = PretrainedVisionTask(device=DEVICE, lr=3e-5, buffer_size=32)
    explorer = RNDExplorer(device=DEVICE, use_ensemble=use_ensemble, vision_task=vision_task,
                           rollout_horizon=rollout_horizon, top_k=top_k, save_freq=save_freq)
    # explorer = RandomExplorer(device=DEVICE, vision_task=vision_task, rollout_horizon=rollout_horizon,
    #                           top_k=top_k, save_freq=save_freq)
    explorer.explore(env, n_steps=length)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--length', type=int, default=1000,
                        help='Number of steps to run the agent')
    parser.add_argument('--width', type=int, default=80,
                        help='Horizontal size of the observations')
    parser.add_argument('--height', type=int, default=80,
                        help='Vertical size of the observations')
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
    run(args.length, args.width, args.height, args.fps, args.level_script,
        args.record, args.demo, args.demofiles, args.video)
