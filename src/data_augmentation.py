import imgaug.augmenters as iaa
import imgaug as ia
import cv2
import os, sys
from shutil import copyfile
import yaml

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
        images = []
        img_filename = []
        for filename in os.listdir(self.path):
            if "png" in filename or "jpg" in filename:
                img = cv2.imread(os.path.join(self.path, filename))
                if img is not None:
                    images.append(img)
                    img_filename.append(filename)
                print("image loded ",str(filename))
        return images, img_filename

    def write_images(self, images, img_filename, num):
        for i in range(0,len(images)):
            if "png" in img_filename[i]:
                img_filename_str = img_filename[i].split(".png")
                cv2.imwrite(self.yaml['save_img_path']+'/'+img_filename_str[0]+"_"+str(num)+".png", images[i])
                origin_txt_path = self.yaml['origin_txt_path']+'/'+img_filename_str[0]+".txt"
                save_txt_path = self.yaml['save_img_path']+'/'+img_filename_str[0]+"_"+str(num)+".txt"
                copyfile(origin_txt_path, save_txt_path)
            elif "jpg" in img_filename[i]:
                img_filename_str = img_filename[i].split(".jpg")
                cv2.imwrite(self.yaml['save_img_path']+'/'+img_filename_str[0]+"_"+str(num)+".jpg", images[i])
                origin_txt_path = self.yaml['origin_txt_path']+'/'+img_filename_str[0]+".txt"
                save_txt_path = self.yaml['save_img_path']+'/'+img_filename_str[0]+"_"+str(num)+".txt"
                copyfile(origin_txt_path, save_txt_path)
        print("image & txt save complete")
        
    def augmentations(self, images):
        
        seq1 = iaa.Sequential([
            iaa.AverageBlur(k=(3,3))
        ])
        seq2 = iaa.ChannelShuffle(p=1.0)
        seq3 = iaa.Dropout((0.05, 0.1), per_channel=0.5)
        seq4 = iaa.Sequential([
            iaa.Add((-10,10)),
            iaa.Multiply((0.7, 1.3))
        ])
        seq5 = iaa.Sequential([
            iaa.Crop(px=(0, 60)),
            iaa.GaussianBlur(sigma=(0, 1.5))
        ])
        print("image augmentation beginning")
        img1=seq1.augment_images(images)
        print("sequence 1 completed......")
        img2=seq2.augment_images(images)
        print("sequence 2 completed......")
        img3=seq3.augment_images(images)
        print("sequence 3 completed......")
        img4=seq4.augment_images(images)
        print("sequence 4 completed......")
        img5=seq5.augment_images(images)
        print("proceed to next augmentations")
        list = [img1, img2, img3, img4, img5]
        return list
    

def main():
    da = DataAugmentation()
    image, image_filename = da.load_images_from_folder()
    photo_augmented = da.augmentations(image)
    for num in range(0, len(photo_augmented)):
        da.write_images(photo_augmented[num], image_filename, num)


if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('Usage: ')
        exit('$ python data_augmentation.py')

    main()