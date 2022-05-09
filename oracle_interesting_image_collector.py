# from pynput import keyboard
import numpy as np
import os
import shutil
import time
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import signal

import deepmind_lab
import site

from agent_random_distill import STEPS_TO_FULLY_ROTATE, LOOKAROUND_ACTION
from tqdm import tqdm

OBSERVATION_MODE = "DEBUG.CAMERA_INTERLEAVED.PLAYER_VIEW_NO_RETICLE"
objects = [chr(c) for c in range(ord("A"), ord("Z")+1) if chr(c) != 'P'][:20]

def _action(*entries):
    return np.array(entries, dtype=np.intc)

ActionSpace = {
    'look_left': _action(-50, 0, 0, 0, 0, 0, 0),
    'look_right': _action(50, 0, 0, 0, 0, 0, 0),
    'strafe_left': _action(0, 0, -1, 0, 0, 0, 0),
    'strafe_right': _action(0, 0, 1, 0, 0, 0, 0),
    'forward': _action(0, 0, 0, 1, 0, 0, 0),
    'backward': _action(0, 0, 0, -1, 0, 0, 0),
    'noop': _action(0, 0, 0, 0, 0, 0, 0),
    'fire': _action(0, 0, 0, 0, -1, 0, 0),
    'jump': _action(0, 0, 0, 0, 0, 1, 0),
    'crouch': _action(0, 0, 0, 0, 0, 0, 1)
}

def on_press(key):
    """
    Callback for keyboard events. Modifies global variables directly.

    :param key: pressed key
    :return:
    """
    global is_capture, env_action, is_done

    if key == keyboard.Key.space:
        is_capture = True

    else:
        try:
            if key.char == "w":
                env_action = dict(action=ActionSpace['forward'], num_steps=10)
            elif key.char == "s":
                env_action =   dict(action=ActionSpace['backward'], num_steps=10)
            elif key.char == "a":
                env_action =   dict(action=ActionSpace['strafe_left'], num_steps=10)
            elif key.char == "d":
                env_action =   dict(action=ActionSpace['strafe_right'] , num_steps=10)
            elif key.char == "h":
                is_done = True
        except Exception as e:
            if key == keyboard.Key.left:
                env_action = dict(action=ActionSpace['look_left'] , num_steps=1)
            elif key == keyboard.Key.right:
                env_action = dict(action=ActionSpace['look_right'] , num_steps=1)


def run(width, height, fps, level):
    """Spins up an environment and runs the random agent."""
    # Extremely hacky way to move custom object/level files to site-packages
    src_root = "../deepmind-lab"
    dst_root = os.path.join(site.getsitepackages()[0], 'deepmind_lab', 'baselab')
    files = [
        f"game_scripts/levels/{level}.lua",
        "game_scripts/common/vlr_objects.lua"
    ]
    for fname in files:
        print(fname)
        src = os.path.join(src_root, fname)
        dst = os.path.join(dst_root, fname)
        print(dst)
        shutil.copyfile(src, dst)

    config = {
        'fps': str(fps),
        'width': str(width),
        'height': str(height)
    }

    # NOTE: RGBD_INTERLEAVED or RGBD also available, potentially train 3D visual features
    # other_states = ["DEBUG.POS.TRANS", "DEBUG.POS.ROT", "DEBUG.CAMERA_INTERLEAVED.PLAYER_VIEW",
    #                 "DEBUG.CAMERA_INTERLEAVED.PLAYER_VIEW_NO_RETICLE"]
    env = deepmind_lab.Lab(level, [OBSERVATION_MODE], config=config)

    img_dir = "vlr_dataset_oracle_interesting_images"

    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)

    global is_capture, env_action, is_done
    is_capture = True
    env_action = None
    is_done = False
    idx = 0
    num_maps = 6 * 20
    image_idx = 0
    while idx < num_maps:
        if idx % 6 == 0:
            obj_id = objects[idx // 6]
            print(f"Processing Object {obj_id}")
            pbar = tqdm(total=6)
            subdir = os.path.join(img_dir, obj_id)
            if not os.path.isdir(subdir):
                os.mkdir(subdir)

        env.reset()

        #### Wait for velocity to reach 0
        for i in range(3 * STEPS_TO_FULLY_ROTATE):
            env.step(LOOKAROUND_ACTION)

        env.step(ActionSpace['noop'], num_steps=3)
        ##########

        ### Capture Images
        for _ in range(3):
            env.step(ActionSpace['forward'], num_steps=4)
            image_idx = sweep_and_save_imgs(env, image_idx, subdir)
        ##########

        pbar.update(1)

        idx += 1
        
        # Close pbar
        if idx % 6 == 0 and idx > 0:
            pbar.close()


def sweep_and_save_imgs(env, image_idx, subdir):
    # Go to extreme left
    env.step(ActionSpace['look_left'], num_steps=6)

    # Save images from 12 different view points
    for _ in range(12):
        env.step(ActionSpace['look_right'], num_steps=1)
        image = env.observations()[OBSERVATION_MODE]
        im = Image.fromarray(image)
        im.save(os.path.join(subdir, f"{image_idx:05d}.png"))
        image_idx += 1

    # Reorient to original angle
    env.step(ActionSpace['look_left'], num_steps=6)
    return image_idx

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width', type=int, default=512,
                        help='Horizontal size of the observations')
    parser.add_argument('--height', type=int, default=512,
                        help='Vertical size of the observations')
    parser.add_argument('--fps', type=int, default=60,
                        help='Number of frames per second')
    parser.add_argument('--level_script', type=str,
                        default='final_basic_map_generator_vlr',
                        help='The environment level script to load')

    args = parser.parse_args()
    run(args.width, args.height, args.fps, args.level_script)
