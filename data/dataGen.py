import sys
import getopt
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import os
import re

def main(argv):
    img_dir = 'tiny_patches/'
    # try:
    #     opts, args = getopt.getopt(argv, "hi:", ["help=", "image="])
    # except getopt.GetoptError:
    #     print ('Please type in the image url')
    #     sys.exit(2)
    # for opt, arg in opts:
    #     if opt == '-h':
    #         print ('dataGen.py -i <image>')
    #         sys.exit()
    #     elif opt in ("-i", "--image"):
    #         img_orig = np.array(Image.open(arg))
    #         symbol = arg[13]
    # print ('image shape : %d x %d' % (img_orig.shape[0], img_orig.shape[1]))
    # img_height, img_width = img_orig.shape

# move a pixel to the left / down
    # for i in range(0, 5):
    #     i = i + 1
    #     for j in range(0, 8):
    #         j = j + 1
    #         img_gen = np.zeros_like(img_orig)
    #         img_gen[:img_height - j, :img_width - i] = img_orig[j:, i:]
    #         img_gen[img_height - j:, img_width - i:] = img_orig[:j, :i]
    #         name_gen = 'tiny_patches/%s%d_%d.jpg' %(symbol, i, j)
    #         scipy.misc.imsave(name_gen, img_gen)
# move a pixel to the right / up
    # for i in range(0, 5):
    #     i += 1
    #     for j in range(0, 8):
    #         j += 1
    #         img_gen = np.zeros_like(img_orig)
    #         img_gen[j:, i:] = img_orig[:img_height - j, :img_width - i]
    #         img_gen[:j, :i] = img_orig[img_height - j:, img_width - i:]
    #         name_gen = 'tiny_patches/%s%d__%d.jpg' %(symbol, i, j)
    #         scipy.misc.imsave(name_gen, img_gen)

# gaussian
    print('performing gaussian blur on the original images...')
    img_names = os.listdir(img_dir)
    is_jpg = re.compile(r'.+?\.jpg')
    for name in img_names:
        if(is_jpg.match(name)):
            image = Image.open(img_dir + name)
            image = image.filter(ImageFilter.GaussianBlur(radius=1)) 
            image.save(img_dir + name.strip('g.jpg') + 'g.jpg')
    print('gaussian blur completed.')
if __name__ == "__main__":
    main(sys.argv[1:])