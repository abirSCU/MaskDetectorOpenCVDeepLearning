import os, cv2
from shutil import copyfile


def main():

  dir_ = './mask_val/2/'
  for entry in os.scandir(dir_):
    if entry.path.endswith('.jpg'):
      img = cv2.imread(entry.path)
      crop_img = img[300:900, 250:750]
      cv2.imwrite('./mask_val/2c/'+ entry.name, crop_img)



if __name__ == '__main__':
  main()