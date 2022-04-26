import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import matplotlib.pyplot as plt

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


def count_diff_SSIM(img1, img2, crop_bounds_w, crop_bounds_h,
        similarity_threshold, verbose=False):
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
    diff_img = (diff_img * 255).astype("uint8")

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff_img, similarity_threshold, 1,
        cv2.THRESH_BINARY_INV)[1]
    if verbose:
        fig = plt.figure()
        print(np.sum(thresh))
        cv2.imshow("New Image", raw_img1)
        cv2.imshow("Prev Image", raw_img2)
        plt.imshow(thresh)
        plt.show()
        cv2.waitKey(1)
    return np.sum(thresh)


def calc_nls_mask(img1, img2, crop_bounds_w, crop_bounds_h,
        maxsp=200, iters=10, verbose=False):
    img1 = img1[crop_bounds_h[0]:crop_bounds_h[1],
                    crop_bounds_w[0]:crop_bounds_w[1]]
    img2 = img2[crop_bounds_h[0]:crop_bounds_h[1],
                    crop_bounds_w[0]:crop_bounds_w[1]]
    image_seq = np.vstack([[img1], [img2]])
    mask = nlc.nlc(image_seq, maxsp=maxsp, iters=iters, outdir='', suffix='',
            clearBlobs=False, binTh=None, relEnergy=None,
            redirect=False, doload=False, dosave=False)
    if verbose:
        fig = plt.figure()
        cv2.imshow("New Image", img1)
        cv2.imshow("Prev Image", img2)
        plt.imshow(mask[1])
        plt.title("NLS Mask for 2nd Image")
        plt.show()
        cv2.waitKey(1)

    return mask


if __name__ == "__main__":
    folder = "test_obj_motion_images"
    im1 = cv2.imread(folder + "/0.png")
    im2 = cv2.imread(folder + "/1.png")
    im3 = cv2.imread(folder + "/2.png")
    crop_bounds_w = [0, im1.shape[1]]
    crop_bounds_h = [0, im1.shape[0]]
    similarity_threshold = 50
    # count_diff_SSIM(im2, im3, crop_bounds_w, crop_bounds_h,
    #     similarity_threshold, verbose=True)

    calc_nls_mask(im1, im2, crop_bounds_w, crop_bounds_h,
        maxsp=200, iters=10, verbose=True)
