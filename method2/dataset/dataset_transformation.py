import xml.etree.ElementTree as ET
import sys, argparse, os
import cv2

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--img', help = "image width and height in px", type = int, default = 400)
  args = parser.parse_args()
  imageNewSize = args.img

  print("Images will be resized to {}x{} pixels".format(imageNewSize, imageNewSize))
  pathToImages = './images'
  for imageName in os.listdir(pathToImages):
    if imageName.endswith('.png'):
      img = cv2.imread(pathToImages + '/' + imageName, cv2.IMREAD_UNCHANGED)
      resizedImg = cv2.resize(img, (imageNewSize, imageNewSize))
      cv2.imwrite(pathToImages + '/' + imageName, resizedImg)
  
  pathToAnnotations = './annotations'
  for annotation in os.listdir(pathToAnnotations):
    tree = ET.parse(pathToAnnotations + '/' + annotation)
    root = tree.getroot()
    for size in root.findall('size'):
      old_width = int(size.find('width').text)
      size.find('width').text = str(imageNewSize)
      old_height = int(size.find('height').text)
      size.find('height').text = str(imageNewSize)
      dw = imageNewSize/old_width
      dh = imageNewSize/old_height
    for object_ in root.findall('object'):
      box = object_.find('bndbox')
      xmin = int(box.find('xmin').text)
      box.find('xmin').text = str(round(xmin*dw))
      ymin = int(box.find('ymin').text)
      box.find('ymin').text = str(round(ymin*dh))
      xmax = int(box.find('xmax').text)
      box.find('xmax').text = str(round(xmax*dw))
      ymax = int(box.find('ymax').text)
      box.find('ymax').text = str(round(ymax*dh))
    tree.write(pathToAnnotations + '/' + annotation)

  




if __name__ == '__main__':
  main()