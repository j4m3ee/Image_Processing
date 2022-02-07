

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import exposure
from skimage.exposure import cumulative_distribution

PATH_IMG = r"G:\CE\Year3\Semester2\ImgPro\scr.jpg"
PATH_IMG_2 = r"G:\CE\Year3\Semester2\ImgPro\src3.jpg"

def cdf(im):
    c, b = cumulative_distribution(im)

    for i in range(b[0]):
        c = np.insert(c, 0, 0)
    for i in range(b[-1]+1, 256):
        c = np.append(c, 1)
    return c

def hist_matching(c, c_t, im):
    b = np.interp(c, c_t, np.arange(256))
    pix_repl = {i:b[i] for i in range(256)}
    mp = np.arange(0, 256)
    for (k, v) in pix_repl.items():
        mp[k] = v
    s = im.shape
    im = np.reshape(mp[im.ravel()], im.shape)
    im = np.reshape(im, s)
    return im


def main():
    set_size = 20
    
    src = cv.imread(PATH_IMG)
    img = cv.cvtColor(src, cv.COLOR_BGR2RGB)
    src2 = cv.imread(PATH_IMG_2)
    img2 = cv.cvtColor(src2, cv.COLOR_BGR2RGB)
    im_result = cv.cvtColor(src, cv.COLOR_BGR2RGB)

    cdf_im1_R = cdf(img[:,:,0])
    cdf_im2_R = cdf(img2[:,:,0])

    cdf_im1_G = cdf(img[:,:,1])
    cdf_im2_G = cdf(img2[:,:,1])

    cdf_im1_B = cdf(img[:,:,2])
    cdf_im2_B = cdf(img2[:,:,2])

    im_result[:,:,0] = hist_matching(cdf_im1_R, cdf_im2_R, img[:,:,0])
    im_result[:,:,1] = hist_matching(cdf_im1_G, cdf_im2_G, img[:,:,1])
    im_result[:,:,2] = hist_matching(cdf_im1_B, cdf_im2_B, img[:,:,2])

    hist1 = cv.calcHist([img[:,:, 0]], [0], None, [256], [0, 256])
    hist11 = cv.calcHist([img[:,:, 1]], [0], None, [256], [0, 256])
    hist111 = cv.calcHist([img[:,:, 2]], [0], None, [256], [0, 256])

    hist2 = cv.calcHist([img2[:,:,0]], [0], None, [256], [0, 256])
    hist22 = cv.calcHist([img2[:,:,1]], [0], None, [256], [0, 256])
    hist222 = cv.calcHist([img2[:,:,2]], [0], None, [256], [0, 256])

    hist_result0 = cv.calcHist([im_result[:,:,0]], [0], None, [256], [0, 256])
    hist_result1 = cv.calcHist([im_result[:,:,1]], [0], None, [256], [0, 256])
    hist_result2 = cv.calcHist([im_result[:,:,2]], [0], None, [256], [0, 256])

    f, ax = plt.subplots(3, 2)
    f.set_figheight(set_size)
    f.set_figwidth(set_size)

    ax[0, 0].imshow(img)
    ax[0, 1].plot(hist1, color = 'r')
    ax[0, 1].plot(hist11, color = 'g')
    ax[0, 1].plot(hist111, color = 'b')
    ax[0, 0].set_title('Image #1')
    ax[0, 1].set_title('Hist Original Image')

    ax[1, 0].imshow(img2)
    ax[1, 1].plot(hist2, color = 'r')
    ax[1, 1].plot(hist22, color = 'g')
    ax[1, 1].plot(hist222, color = 'b')
    ax[1, 0].set_title('Image #2')
    ax[1, 1].set_title('Hist Equalized Image')

    ax[2, 0].imshow(im_result)
    ax[2, 1].plot(hist_result0, color = 'r')
    ax[2, 1].plot(hist_result1, color = 'g')
    ax[2, 1].plot(hist_result2, color = 'b')
    ax[2, 0].set_title('Hist Matching Image')
    ax[2, 1].set_title('Hist Equalized Image')

    plt.savefig('act3.3.png')
    plt.show()

main()