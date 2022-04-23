import numpy as np
import matplotlib.pyplot as plt
import signal


def sigint_handler(signal, frame):
    # Force scripts to exit cleanly
    exit()


signal.signal(signal.SIGINT, sigint_handler)

images = np.load('experiment_data/images.npy')
action_distribs = np.load('experiment_data/action_distribs.npy')
num_actions = len(action_distribs[0])
angles = (360 * np.arange(num_actions) / num_actions).astype(int)

f, axes = plt.subplots(1, 7, figsize=(40, 5))
train_iter = 0
for i in range(0, len(images), 5):
    for j in range(len(images[i])):
        axes[j].imshow(images[i][j])
        axes[j].set_title("(%d: %.2f)" % (angles[j], action_distribs[i][j]))
    train_iter = int(i / 16)
    f.suptitle('Train Iteration: {}, step: {}'.format(train_iter, i))
    plt.draw()
    plt.waitforbuttonpress()
    for j in range(len(images[i])):
        axes[j].clear()
