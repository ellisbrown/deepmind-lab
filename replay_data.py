import numpy as np
import matplotlib.pyplot as plt
import signal


def sigint_handler(signal, frame):
    # Force scripts to exit cleanly
    exit()


signal.signal(signal.SIGINT, sigint_handler)

# Initial RND:
# data_root = """
# experiment_data/initial_RND
# """

# Initial Random:
data_root = """
experiment_data/initial_random
"""
data_root = data_root.strip()

interesting_images = np.load(f'{data_root}/interesting_images.npy', allow_pickle=True)
print(len(interesting_images))
for i, im in enumerate(interesting_images):
    print("Image {}".format(i))
    plt.imshow(im)
    plt.show()

exit()

images = np.load(f'{data_root}/all_images.npy')
action_distribs = np.load(f'{data_root}/all_action_distribs.npy')
intrinsic_rewards = np.load(f'{data_root}/all_intrinsic_reward.npy', allow_pickle=True)
num_actions = len(action_distribs[0])
angles = (360 * np.arange(num_actions) / num_actions).astype(int)
f, axes = plt.subplots(1, 7, figsize=(40, 5))
train_iter = 0
for i in range(0, len(images), 5):
    for j in range(len(images[i])):
        axes[j].imshow(images[i][j])
        axes[j].set_title("(%d: %.2f, %.2f)" % (angles[j], action_distribs[i][j], intrinsic_rewards[i][j]))
    train_iter = int(i / 16)
    f.suptitle('Train Iteration: {}, step: {}'.format(train_iter, i))
    plt.draw()
    plt.waitforbuttonpress()
    for j in range(len(images[i])):
        axes[j].clear()
