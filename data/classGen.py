import os
import sys

def main(argv):
    img_dir = "tiny_patches/"
    img_list = []  
    img_names = os.listdir(img_dir)
    file=open('classes.txt','w')  
    if (len(img_names)>0):  
        for fn in img_names:
            file.write(fn[0])
            file.write('\n')
if __name__ == "__main__":
    main(sys.argv[1:])