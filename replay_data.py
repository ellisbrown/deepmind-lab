import numpy as np
import matplotlib.pyplot as plt
import signal
import os
import ipdb

def sigint_handler(signal, frame):
    # Force scripts to exit cleanly
    exit()


signal.signal(signal.SIGINT, sigint_handler)

def draw_vec(pos: np.ndarray, theta_rad: float, radius: float, color: str, width=0.025, **kwargs):
    """
    Draws a vector from pos to pos + radius * [cos(theta_rad), sin(theta_rad)]
    """
    arrow_len = radius
    x0, y0 = pos
    plt.arrow(x0, y0, arrow_len * np.cos(theta_rad), arrow_len * np.sin(theta_rad),
              color=color, length_includes_head=True, width=width, shape='full', **kwargs)


data_root = """
experiment_data/2022_04_30_22:53:16
"""
data_root = data_root.strip()
if not os.path.exists(f"{data_root}/extracted_images"):
    os.mkdir(f"{data_root}/extracted_images")

# interesting_images = np.load(f'{data_root}/interesting_images.npy', allow_pickle=True)
# print(len(interesting_images))
# for i in range(len(interesting_images)):
#     if i % 2 == 0:  # trained every group of 4 images so visualize every 4 training iters
#         im = interesting_images[i]
#         print("Image {}/{}".format(i, len(interesting_images)))
#         cv2.imshow("Image %d/%d" % (i, len(interesting_images)), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
#         plt.imsave(f"{data_root}/extracted_images/im_{i}_{len(interesting_images)}.png", im)
#         if cv2.waitKey(0) == ord("q"):
#             exit()
#         cv2.destroyAllWindows()
#
# exit()


# states = np.load(f'{data_root}/all_states.npy')
# start_pos = states[0, :2]
# min_xy = np.min(states[:, :2] - start_pos[np.newaxis], axis=0)
# max_xy = np.max(states[:, :2] - start_pos[np.newaxis], axis=0)
# fig, ax = plt.subplots(figsize=(10, 5))
# for i in range(len(states)):
#     ax.clear()
#     # ax.set_title(title)
#
#     # Draw fixed object positions
#     # ax.add_patch(plt.Circle(cur_pos, color=agent_color, radius=agent_radius, alpha=0.4, label="Agent"))
#
#     pos, theta = states[i, :2], states[i, 4]
#     draw_vec(pos=pos - start_pos, theta_rad=theta, radius=3, color="black")
#
#     ax.set_xlim(min_xy[0], max_xy[0])
#     ax.set_ylim(min_xy[1], max_xy[1])
#     ax.set_xbound(min_xy[0], max_xy[0])
#     ax.set_xbound(min_xy[1], max_xy[1])
#     plt.draw()
#     plt.pause(0.3)  # Give enough time to plot before ipdb

images = np.load(f'{data_root}/all_images.npy')
all_actions = np.load(f'{data_root}/all_actions.npy')
try:
    action_distribs = np.load(f'{data_root}/all_action_distribs.npy')
except FileNotFoundError:
    action_distribs1 = np.load(f'{data_root}/all_action_distribs1.npy')
    action_distribs2 = np.load(f'{data_root}/all_action_distribs2.npy')

    action_distribs = action_distribs2

intrinsic_rewards = np.load(f'{data_root}/all_intrinsic_reward.npy', allow_pickle=True)
num_actions = len(action_distribs[0])
angles = (360 * np.arange(num_actions) / num_actions).astype(int)
f, axes = plt.subplots(1, num_actions, figsize=(40, 5))
train_iter = 0
for i in range(0, len(images), 8):
    chosen_idx = all_actions[i]
    # try:
    #     assert chosen_idx == np.argmax(intrinsic_rewards[i])
    # except:
    #     print("Chosen action:", chosen_idx, " not matching Intrinsic reward:", intrinsic_rewards[i])
    # import ipdb
    # ipdb.set_trace()
    image_order = (np.arange(num_actions) - chosen_idx + 3) % num_actions  # make chosen_idx the 3rd(middle) image always
    for j, order_idx in enumerate(image_order):
        axes[order_idx].imshow(images[i][j])
        # axes[order_idx].set_title("%d" % (intrinsic_rewards[i][j]), fontsize=20)
        axes[order_idx].set_title("%d" % (action_distribs[i][j] * 100), fontsize=20)
        axes[order_idx].axis("off")
    train_iter = int(i / 16)
    f.suptitle('Train Iteration: {}, step: {}'.format(train_iter, i))
    plt.tight_layout()
    plt.draw()
    plt.waitforbuttonpress()
    for j in range(len(images[i])):
        axes[j].clear()
