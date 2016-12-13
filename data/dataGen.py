import sys
import getopt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hi:", ["help=", "image="])
    except getopt.GetoptError:
        print ('Please type in the image url')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('dataGen.py -i <image>')
            sys.exit()
        elif opt in ("-i", "--image"):
            img_orig = np.array(Image.open(arg))
            symbol = arg[13]
    print ('image shape : %d x %d' % (img_orig.shape[0], img_orig.shape[1]))
    img_height, img_width = img_orig.shape

    for i in range(0, 5):
        i = i + 1
        for j in range(0, 8):
            j = j + 1
            img_gen = np.zeros_like(img_orig)
            img_gen[:img_height - j, :img_width - i] = img_orig[j:, i:]
            img_gen[img_height - j:, img_width - i:] = img_orig[:j, :i]
            name_gen = 'tiny_patches/%s%d_%d.jpg' %(symbol, i, j)
            scipy.misc.imsave(name_gen, img_gen)

    for i in range(0, 5):
        i += 1
        for j in range(0, 8):
            j += 1
            img_gen = np.zeros_like(img_orig)
            img_gen[j:, i:] = img_orig[:img_height - j, :img_width - i]
            img_gen[:j, :i] = img_orig[img_height - j:, img_width - i:]
            name_gen = 'tiny_patches/%s%d__%d.jpg' %(symbol, i, j)
            scipy.misc.imsave(name_gen, img_gen)

    # gaussian 

if __name__ == "__main__":
    main(sys.argv[1:])