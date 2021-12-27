import albumentations as A
import random
import cv2
import os, sys
from shutil import copyfile
import yaml
import numpy as np
import multiprocessing

class DataAugmentation():
    def __init__(self):
        ### load param
        try:
            config_name = './param/' + 'data_augmentation.yaml'
            with open(config_name) as file:
                self.yaml = yaml.load(file, Loader=yaml.FullLoader)
        except:
            exit('ERROR: data_augmentation.yaml not defined.') 
        
        self.path = self.yaml['origin_img_path']
    
    def load_images_from_folder(self):
        file_path = []
        file_name = []
        for filename in os.listdir(self.path):
            if self.yaml['file_ext'] in filename:
                file_path.append(self.path+'/'+filename[:-4])
                file_name.append(filename)
        return file_path, file_name

    def write_images(self, augmented_data, img_filename, num):
        if self.yaml['file_ext'] in img_filename:
            img_filename_str = img_filename.split("."+self.yaml['file_ext'])
            cv2.imwrite(self.yaml['save_img_path']+'/'+img_filename_str[0]+"_"+str(num)+"."+self.yaml['file_ext'], augmented_data['image'])
            # origin_txt_path = self.yaml['origin_txt_path']+'/'+img_filename_str[0]+".txt"
            save_txt_path = self.yaml['save_img_path']+'/'+img_filename_str[0]+"_"+str(num)+".txt"
            # print(img_filename_str)
            f = open(save_txt_path,'w')
            txt_data = []
            # lines = .readlines()
            for i in range(0, len(augmented_data['bboxes'])):
                pre_data = str(augmented_data['bboxes'][i]).split(" ")
                # print(pre_data)
                pre_data[4] = pre_data[4].split(")")[0]
                pre_data[0] = pre_data[0].split("(")[1].split(",")[0]
                pre_data[1] = pre_data[1].split(",")[0]
                pre_data[2] = pre_data[2].split(",")[0]
                pre_data[3] = pre_data[3].split(",")[0]
                for j in range(0, len(pre_data)-1):
                    if float(pre_data[j]) <= 0.0:
                        pre_data[j] = "0"
                    elif float(pre_data[j]) >= 1.0:
                        pre_data[j] = "1"
                # print(pre_data[4]+" "+pre_data[0]+" "+pre_data[1]+" "+pre_data[2]+" "+pre_data[3])
                txt_data.append(pre_data[4]+" "+
                                pre_data[0]+" "+
                                pre_data[1]+" "+
                                pre_data[2]+" "+
                                pre_data[3]+"\n")
            f.writelines(txt_data)
            f.close()
            # print(save_txt_path)
        # print("image & txt save complete")
        
    def augmentations(self, file_path):
        
        image = cv2.imread(file_path+"."+self.yaml['file_ext'])
        f = open(file_path+'.txt','r')
        lines = f.readlines()
        bboxes = []
        for i in range(0, len(lines)):
            pre_data = lines[i].split(" ")
            bbox = []
            for j in range(5): # 0 1 2 3 4
                n = j+1 # 1 2 3 4 5
                if j is 4:
                    n = 0
                    bbox.append(int(pre_data[n]))
                else:
                    bbox.append(float(pre_data[n]))
            bboxes.append(bbox)
        f.close()
        
        transforms = A.Compose([
            # A.Resize(960,1280),
            A.OneOf([
                A.RandomSizedBBoxSafeCrop(height = 480,width = 640,erosion_rate =  0.0, interpolation = 1,p=1),
                A.RandomSizedBBoxSafeCrop(height = 960,width = 1280,erosion_rate =  0.0, interpolation = 1,p=1),
                # A.RandomSizedBBoxSafeCrop(height = 1440,width = 1920,erosion_rate =  0.0, interpolation = 1,p=0.7),
                # A.RandomSizedBBoxSafeCrop(height = 1920,width = 2560,erosion_rate =  0.0, interpolation = 1,p=0.7),
                # A.RandomSizedBBoxSafeCrop(height = 2400,width = 3200,erosion_rate =  0.0, interpolation = 1,p=0.7),
            ]),
            A.Rotate (limit=15, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.2),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=(0,15), val_shift_limit=15, always_apply=False, p=0.7),
            A.Blur(blur_limit=2, always_apply=False, p=0.5),
            A.CLAHE (clip_limit=1, tile_grid_size=(8, 8), always_apply=False, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=[-0.1,0.01], contrast_limit=[0,0.3],p=0.9),
            A.RandomShadow (shadow_roi=(0.3, 0.3, 1, 1), num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, always_apply=False, p=0.6),
            A.RandomSunFlare (flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, num_flare_circles_lower=0, num_flare_circles_upper=2, src_radius=50, src_color=(255, 255, 255), always_apply=False, p=0.5)
            ],
            bbox_params=A.BboxParams(format="yolo"))
        data = transforms(image=image, bboxes=bboxes)
        
        return data
    

def main():
    da = DataAugmentation()
    file_path, file_name = da.load_images_from_folder()

    for i in range(len(file_path)):
        for j in range(8):
            augmented_data = da.augmentations(file_path[i])
            da.write_images(augmented_data, file_name[i], j)
        print(str(i)+"/"+str(len(file_path)))

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Usage: ')
        exit('$ python data_augmentation.py')
    main()