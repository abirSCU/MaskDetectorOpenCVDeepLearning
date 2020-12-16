import glob, os, random
from shutil import copyfile
import xml.etree.ElementTree as ET


def main():
  
  images_dir = './images/'
  percentage_test = 0.1
  percentage_val = 0.2
  
  all_images = os.listdir(images_dir)
  n_images = len(all_images)
  test_list = random.sample(all_images, k=round(n_images*percentage_test))
  all_images = list(set(all_images).difference(set(test_list)))
  val_list = random.sample(all_images, k=round(n_images*percentage_val))
  train_list = list(set(all_images).difference(set(val_list)))

  lists_dictionary = {'test': test_list, 'train': train_list, 'valid': val_list}

  output_path = '../yolov5/face_masks_data/'
  os.mkdir(output_path)
  for type_ in ['test', 'train', 'valid']:
    os.mkdir(output_path + type_)
    os.mkdir(output_path + '{}/images'.format(type_))
    os.mkdir(output_path + '{}/annotations'.format(type_))
    for i in lists_dictionary[type_]:
      copyfile(images_dir + i, output_path + '{}/images/'.format(type_) + i)

  annotations_dir = './annotations'
  for filename in os.listdir(annotations_dir):
    if filename.endswith('.xml'):
      for type_ in ['test', 'train', 'valid']:
        if filename.replace('.xml','.png') in lists_dictionary[type_]:
          tree = ET.parse(annotations_dir + '/' + filename)
          root = tree.getroot()
          for size in root.findall('size'):
            picture_width = int(size.find('width').text)
            picture_heigth = int(size.find('height').text)
          labels = []
          xMins = []
          xMaxs = []
          yMins = []
          yMaxs = []
          dw = 1./picture_width
          dh = 1./picture_heigth
          i = 0

          txtName = filename.replace('.xml', '.txt')
          outF = open(output_path + type_ + '/annotations/' + txtName, "w")
          for object_ in root.findall('object'):
            tmpLabel = object_.find('name').text
            if tmpLabel == 'without_mask':
              tmpLabel = '0'
            elif tmpLabel == 'mask_weared_incorrect':
              tmpLabel = '1'
            elif tmpLabel == 'with_mask':
              tmpLabel = '2'
            labels.append(tmpLabel)
            box = object_.find('bndbox')
            xMins.append(int(box.find('xmin').text))
            yMins.append(int(box.find('ymin').text))
            xMaxs.append(int(box.find('xmax').text))
            yMaxs.append(int(box.find('ymax').text))
            
            x = (xMaxs[i] + xMins[i])/2.0
            y = (yMaxs[i] + yMins[i])/2.0
            width = xMaxs[i] - xMins[i]
            height = yMaxs[i] - yMins[i]
            x = x*dw
            y = y*dh
            width = width*dw
            height = height*dh

            line = labels[i] + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' '  + str(height)
            outF.write(line)
            outF.write('\n')
            i += 1
          outF.close()
          break
  
  yaml = open(output_path + 'data.yaml' , "w")
  yaml.write('train: train/images\n')
  yaml.write('val: valid/images\n')
  yaml.write('\n')
  yaml.write('nc: 3\n')
  yaml.write("names: ['without_mask', 'mask_weared_incorrect', 'with_mask']\n")
  yaml.close()
        

  


if __name__ == '__main__':
  main()

