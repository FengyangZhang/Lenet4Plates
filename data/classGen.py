import os
import sys
import re

def main(argv):
    img_dir = "tiny_patches/"
    img_list = []  
    img_names = os.listdir(img_dir)
    file=open('classes.txt','w')
    is_jpg = re.compile(r'.+?\.jpg')
    print('generating class labels...') 
    if (len(img_names)>0):  
        for name in img_names:
            if (is_jpg.match(name)):
                file.write(name[0])
                file.write('\t')
    print('class label generated.')
if __name__ == "__main__":
    main(sys.argv[1:])