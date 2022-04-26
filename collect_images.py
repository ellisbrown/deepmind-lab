from pynput import keyboard
import numpy as np
import os
import shutil
import time
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import signal

import deepmind_lab

from agent_random_distill import STEPS_TO_FULLY_ROTATE, LOOKAROUND_ACTION

OBSERVATION_MODE = "DEBUG.CAMERA_INTERLEAVED.PLAYER_VIEW_NO_RETICLE"

def _action(*entries):
    return np.array(entries, dtype=np.intc)


ActionSpace = {
    'look_left': _action(-200, 0, 0, 0, 0, 0, 0),
    'look_right': _action(200, 0, 0, 0, 0, 0, 0),
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

    img_dir = "test_obj_motion_images"
    images = []
    global is_capture, env_action, is_done
    is_capture = False
    env_action = None
    is_done = False
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    def sigint_handler(signal, frame):
        # Force scripts to exit cleanly
        global is_done
        is_done = True
    signal.signal(signal.SIGINT, sigint_handler)

    while not is_done:
        image = env.observations()[OBSERVATION_MODE]
        try:
            plt.imshow(image)
            plt.draw()
            plt.pause(0.01)
            if is_capture:
                # Wait for velocity to reach 0
                for i in range(3 * STEPS_TO_FULLY_ROTATE):
                    env.step(LOOKAROUND_ACTION)

                env.step(ActionSpace['noop'], num_steps=3)
                print("Capturing image")
                image = env.observations()[OBSERVATION_MODE]
                plt.imshow(image)
                plt.draw()
                plt.pause(0.01)
                images.append(image)
                is_capture = False

            elif env_action is not None:
                env.step(**env_action)
                env_action = None

            else:
                pass
        except KeyboardInterrupt:
            break

    num_im = len(os.listdir(img_dir))
    for i, im_arr in enumerate(images):
        print("Saved image {}".format(num_im + i))
        im = Image.fromarray(im_arr)
        im.save(os.path.join(img_dir, f"{num_im + i}.png"))

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
