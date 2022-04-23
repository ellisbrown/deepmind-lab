import argparse
import random
import numpy as np
import six
import matplotlib.pyplot as plt
import os
import shutil
from tqdm import tqdm
from datetime import datetime

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


class RNDExplorer(object):
    """
    Ideas for generating actions

    """

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.lr = 1e-3
        self.rollout_horizon = 16
        self.n_train = 10
        self.eps = 0.0  # p(random action)
        self.num_steps_forward = 10

        # [0(initial view), 1, ... STEPS_TO_FULLY_ROTATE-1(final view before returning to inital view)]
        # takes STEPS_TO_FULLY_ROTATE to full rotate, so to avoid double-counting
        # initial rotation,
        self.action_space = np.arange(STEPS_TO_FULLY_ROTATE)
        self.num_actions = len(self.action_space)

        # TODO: Use ensemble? predictor would minimize average error across ensemble
        #   but seems possible certain states would have a lower bound on error
        #   no matter how many times they are trained on
        self.target = models.resnet18(pretrained=True).to(self.device)
        self.predictor = models.resnet18(pretrained=True).to(self.device)
        for param in self.predictor.parameters():
            param.requires_grad = False
        # NOTE: define model.fc AFTER so only fc has requires_grad=True
        self.optimizer = None
        self.reset_model_weights()

        self.im_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.im_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        self.all_images = []
        self.all_action_distribs = []
        self.all_actions = []
        self.interesting_images = []

    def reset_model_weights(self):
        # Reset predictor and target network final FC layer weights
        # Freeze all target network weights
        self.target.fc = nn.Linear(512, self.num_actions).to(self.device)
        self.predictor.fc = nn.Linear(512, self.num_actions).to(self.device)
        params_to_update = []
        for p in self.predictor.parameters():
            if p.requires_grad == True:  # only fc params will have requires_grad=True
                params_to_update.append(p)

        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(params_to_update, lr=self.lr)

    def update(self, batch):
        init_loss = None
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

    def get_obs(self, env, transform=True):
        if transform:
            import ipdb
            ipdb.set_trace()
            return apply_transforms(env.observations()[OBSERVATION_MODE]).to(self.device)
        else:
            return env.observations()[OBSERVATION_MODE]

    @staticmethod
    def obs_to_im(obs):
        return save_im_resize(obs * im_std.view(3, 1, 1) + im_mean.view(3, 1, 1)
                                   ).permute(1, 2, 0).detach().cpu().numpy()

    def get_action_distrib(self, env):
        images = [self.get_obs(env)]
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
        # print(env.observations()["DEBUG.POS.TRANS"])
        # print(env.observations()["DEBUG.POS.ROT"])

        input_images = torch.stack(images).to(self.device)
        intrinsic_reward = torch.square(self.predictor(input_images) -
                                        self.target(input_images)).sum(dim=-1)
        action_distrib = intrinsic_reward / intrinsic_reward.sum()
        action_distrib = action_distrib.detach().cpu().numpy()
        return action_distrib, images

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

    def rollout(self, env):
        replay_buffer = [self.get_obs(env)]

        # Take actions while collecting images for RND training
        for k in range(self.rollout_horizon):
            # cur_state = self.obs_to_im(self.get_obs(env))
            # plt.imshow(cur_state)
            # plt.show()

            with torch.no_grad():
                # batch x output_feat -> batch x 1
                action_distrib, images = self.get_action_distrib(env)

            # TODO: epsilon greedy? or just sample from action distrib?
            # if np.random.uniform() < self.eps:
            #     # random action
            #     action = np.random.choice(self.action_space)
            # else:
            #     # greedy action
            #     action = np.argmax(self.action_space)
            direction_action = np.random.choice(self.action_space, p=action_distrib)

            # cur_state_again = self.obs_to_im(self.get_obs(env))
            # plt.imshow(cur_state_again)
            # plt.show()

            # print("desired:")
            # assert len(self.action_space) == len(images)
            # desired_im = self.obs_to_im(images[direction_action])
            # plt.imshow(desired_im)
            # plt.show()

            self.take_action(env, direction_action)
            # print("Actions: ", self.action_space)
            # print("Action distrib: ", np.array2string(action_distrib, precision=2))

            if not env.is_running():
                # Env stopped??? Why?
                import ipdb
                ipdb.set_trace()
                break

            replay_buffer.append(self.get_obs(env))
            try:
                images = [self.obs_to_im(obs) for obs in images]
                self.all_images.append(images)
                self.all_action_distribs.append(action_distrib)
                self.all_actions.append(direction_action)
                # plt.imshow(im)
                # plt.draw()
                # plt.pause(0.2)
            except KeyboardInterrupt:
                exit()

        return replay_buffer

    def explore(self, env, max_eps_len):
        # train
        # while not finished_exploring:

        # spin around to get rid of initial "halo" around agent
        self.get_action_distrib(env)
        self.get_action_distrib(env)
        self.get_action_distrib(env)

        step = 0
        pbar = tqdm(total=max_eps_len)
        while step < max_eps_len:
            # NOTE: Reset Replay Buffer to only contain recently visited states
            replay_buffer = self.rollout(env)
            step += len(replay_buffer)
            pbar.update(len(replay_buffer))

            # Update Predictor Network n_train times using the same replay buffer
            batch_states = torch.stack(replay_buffer)
            init_loss, final_loss = self.update(batch_states)
            print("(%d) %.3f -> %.3f" % (step + 1, init_loss, final_loss))

        now = datetime.now()
        now_str = now.strftime("%Y_%m_%d_%H:%M:%S")
        images_dir = "experiment_data/%s" % now_str
        os.mkdir(images_dir)
        np.save(f"{images_dir}/all_images.npy", self.all_images)
        np.save(f"{images_dir}/all_action_distribs.npy", self.all_action_distribs)
        np.save(f"{images_dir}/all_actions.npy", self.all_actions)
        np.save(f"{images_dir}/interesting_images.npy", self.interesting_images)
        print("Saved to %s" % images_dir)
        return

    def __repr__(self):
        return 'RND_Explorer()'


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
    env = deepmind_lab.Lab(level, [OBSERVATION_MODE], config=config)
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

    explorer = RNDExplorer(device=DEVICE)
    explorer.explore(env, max_eps_len=length)


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
