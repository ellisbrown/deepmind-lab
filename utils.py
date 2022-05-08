import numpy as np
import torch
import cv2
import time
from skimage.metrics import structural_similarity as ssim
from PIL import Image

import nlc

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, np.ndarray):
            self.val = np.mean(val)
            self.sum += np.sum(val) * n
            self.count += val.size * n
        else:
            self.val = val
            self.sum += val * n
            self.count += n

        self.avg = self.sum / self.count


def show_image(img, title="Image"):
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)

    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()

    assert img.shape[-1] in [1, 3], "Image must be either grayscale or RGB, not {}".format(img.shape[-1])
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # normalize to 0, 255
    img = (img - np.min(img, axis=(0, 1))) / np.max(img, axis=(0, 1))
    img = (255 * img).astype(np.uint8)
    cv2.imshow(title, img)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)
    cv2.destroyAllWindows()


def calc_SSIM_mask(img1, img2, crop_bounds_w=None, crop_bounds_h=None,
        similarity_threshold=50, verbose=False):
    if crop_bounds_w is None:
        crop_bounds_w = [0, img1.shape[1]]
    if crop_bounds_h is None:
        crop_bounds_h = [0, img1.shape[0]]
    # crop image to only show target
    raw_img1 = img1[crop_bounds_h[0]:crop_bounds_h[1],
                    crop_bounds_w[0]:crop_bounds_w[1]]
    raw_img2 = img2[crop_bounds_h[0]:crop_bounds_h[1],
                    crop_bounds_w[0]:crop_bounds_w[1]]

    # convert the images to grayscale
    gray_img1 = cv2.cvtColor(raw_img1, cv2.COLOR_BGR2GRAY)
    gray_img2 = cv2.cvtColor(raw_img2, cv2.COLOR_BGR2GRAY)

    # compute Structural Similarity Index of the two images
    (score, diff_img) = ssim(gray_img1, gray_img2, full=True)
    diff_img = 255 - (diff_img * 255).astype("uint8")

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    # set value = 1 if difference is above threshold
    # else set value = 0
    thresh = cv2.threshold(diff_img, similarity_threshold, 1, cv2.THRESH_BINARY)[1]
    if verbose:
        print(np.sum(thresh))
        cv2.imshow("New Image", raw_img1)
        cv2.imshow("Prev Image", raw_img2)
        show_image(diff_img)
        cv2.waitKey(1)
    return thresh, diff_img


def calc_nls_mask(img1, img2, crop_bounds_w=None, crop_bounds_h=None,
        maxsp=200, iters=10, verbose=False):
    if crop_bounds_w is None:
        crop_bounds_w = [0, img1.shape[1]]
    if crop_bounds_h is None:
        crop_bounds_h = [0, img1.shape[0]]
    img1 = img1[crop_bounds_h[0]:crop_bounds_h[1],
                    crop_bounds_w[0]:crop_bounds_w[1]]
    img2 = img2[crop_bounds_h[0]:crop_bounds_h[1],
                    crop_bounds_w[0]:crop_bounds_w[1]]
    image_seq = np.vstack([[img1], [img1]])
    mask = nlc.nlc(image_seq, maxsp=maxsp, iters=iters, outdir='', suffix='',
            clearBlobs=False, binTh=None, relEnergy=None,
            redirect=False, doload=False, dosave=False)
    print(np.min(mask[1]), np.max(mask[1]))
    if verbose:
        cv2.imshow("New Image", img1)
        cv2.imshow("Prev Image", img2)
        cv2.imshow("NLS Mask for 2nd Image", (255 * (mask[1] / np.min(mask[1]))).astype(np.uint8))  # 255 * (mask[1] > 0.05)
        np.save("nls_mask.npy", mask)
        cv2.waitKey(0)

    return mask


def get_random_regions(image_RGB, min_area = 50, num_regions=30):
    # image_RGB: (H, W, 3)
    # returns [[left, top, right, bot], ...]
    H, W, _ = image_RGB.shape
    rand_left = np.random.randint(0, W, num_regions)
    rand_width = np.random.randint(0, W, num_regions)
    rand_right = np.clip(rand_left+rand_width, 0, W)

    rand_top = np.random.randint(0, H, num_regions)
    rand_height = np.random.randint(0, H, num_regions)
    rand_bot = np.clip(rand_top+rand_height, 0, H)

    area = (rand_right - rand_left) * (rand_bot - rand_top)
    bboxes = np.array([rand_left, rand_top, rand_right, rand_bot]).T
    bboxes = bboxes[np.where(area > min_area)[0]]
    return bboxes


# TODO: calculate the intersection over union of two boxes
def calc_iou(box1, box2):
    """
    Calculates Intersection over Union for two arrays of bounding boxes
        (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    xmin1, ymin1, xmax1, ymax1 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    xmin2, ymin2, xmax2, ymax2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Calculate the intersection
    x_low = torch.maximum(xmin1, xmin2)
    y_low = torch.maximum(ymin1, ymin2)
    x_high = torch.minimum(xmax1, xmax2)
    y_high = torch.minimum(ymax1, ymax2)

    # y_diff or x_diff are negative if no overlap at all
    y_diff = y_high - y_low
    x_diff = x_high - x_low
    area_intersect = y_diff * x_diff
    invalid_idxs = torch.where((y_diff < 0) | (x_diff < 0))
    area_intersect[invalid_idxs] = 0

    # Calculate the union
    area_box1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area_box2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    area_union = area_box1 + area_box2 - area_intersect

    return area_intersect / area_union


# TODO: given bounding boxes and corresponding scores, perform non max suppression
def calc_nms(bounding_boxes, confidence_score, iou_thresh=0.3):
    """
    TODO: iou_thresh=0.5 for the map of 0.13 stated by handout
    bounding boxes of shape     Nx4
    confidence scores of shape  N
    iou_thresh: two bounding boxes are considered "overlapping" and one is
        pruned if IOU > iou_thresh. As iou_thresh decreases, more boxes are pruned

    return: np.array of bounding boxes and scores sorted in decreasing
            order of confidence score
    """
    # valid_idxs = torch.where(confidence_score > nms_thresh)[0]
    # confidence_score = confidence_score[valid_idxs]
    # bounding_boxes = bounding_boxes[valid_idxs]

    # sort by confidence score
    sorted_idxs = torch.argsort(confidence_score, descending=True)
    confidence_score = confidence_score[sorted_idxs]
    bounding_boxes = bounding_boxes[sorted_idxs]

    # Loop:
    # Algorithm:
    # https://www.analyticsvidhya.com/blog/2020/08/selecting-the-right-bounding-box-using-non-max-suppression-with-implementation/
    remaining_idxs = torch.arange(0, bounding_boxes.shape[0])
    final_boxes = []
    final_scores = []
    while remaining_idxs.shape[0] > 0:
        # get the index of the highest confidence score
        highest_idx = 0
        highest_box = bounding_boxes[highest_idx]
        highest_score = confidence_score[highest_idx]

        # keep this box
        final_boxes.append(highest_box.tolist())
        final_scores.append(highest_score.item())

        # get the indices of the boxes that are within the iou threshold
        iou_idxs = torch.where(calc_iou(highest_box.unsqueeze(0).repeat(bounding_boxes.shape[0], 1),
                                        bounding_boxes) < iou_thresh)[0]
        bounding_boxes = bounding_boxes[iou_idxs]
        confidence_score = confidence_score[iou_idxs]
        remaining_idxs = torch.arange(0, bounding_boxes.shape[0])

    return np.array(final_boxes), np.array(final_scores)


@torch.no_grad()
def get_rpn_regions(rpn, image_RGB, iou_thresh=0.25, min_area=2500):
    """
    min_area: default 2500, at least 50 x 50 region
    """
    # image_RGB: (H, W, 3)
    # returns [[left, top, right, bot], ...]
    H, W, _ = image_RGB.shape
    image_BGR = image_RGB[:, :, ::-1]  # Expects BGR as input
    proposals = rpn(image_BGR)
    bboxes = proposals["proposals"].proposal_boxes.tensor
    scores = proposals["proposals"].objectness_logits

    # First filter out only the top 50% boxes based on scores
    bboxes = bboxes[0:len(bboxes)//2]
    scores = scores[0:len(scores)//2]

    # Second filter out any boxes that are too small
    xmin, ymin, xmax, ymax = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    areas = (xmax - xmin) * (ymax - ymin)
    valid_idxs = torch.where(areas > min_area)[0]
    bboxes = bboxes[valid_idxs]
    scores = scores[valid_idxs]

    bboxes, scores = calc_nms(bboxes, scores, iou_thresh=iou_thresh)

    # make boxes ints and clip them
    bboxes = np.clip(bboxes, [0, 0, 0, 0], [W-1, H-1, W-1, H-1]).astype(np.int)
    return bboxes



if __name__ == "__main__":
    folder = "test_obj_motion_images"
    im1 = np.array(Image.open(folder + "/0.png"))
    im2 = np.array(Image.open(folder + "/1.png"))
    im3 = np.array(Image.open(folder + "/2.png"))
    crop_bounds_w = [0, im1.shape[1]]
    crop_bounds_h = [0, im1.shape[0]]
    similarity_threshold = 50
    # count_diff_SSIM(im2, im3, crop_bounds_w, crop_bounds_h,
    #     similarity_threshold, verbose=True)

    calc_nls_mask(im1, im2, crop_bounds_w, crop_bounds_h,
        maxsp=200, iters=10, verbose=True)

    # plt.imshow(np.load("nls_mask.npy")[1])
    # plt.show()
