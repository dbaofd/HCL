import os
import torch.utils.data as data
import cv2
from data.training_transforms import *

class TrainDataset(data.Dataset):
    # Adopted from https://github.com/janghyuncho/PiCIE/tree/master/data
    # Dataset preparation for unsupervised training.
    def __init__(self, dataset_name, root_path, split, inv_list=[], eqv_list=[], res1=225,
                 res2=450, scale=(0.5, 1)):
        self.res1 = res1
        self.res2 = res2
        self.scale = scale  # eqv scale
        self.inv_list = inv_list
        self.eqv_list = eqv_list
        self.valid_dataset = ["coco", "coco_10k", "coco_iic_subset_train", "pascal", "cityscapes", "imagenet_100", "cityscapes",
                              "coco10k&pascal"]
        self.dataset_name = dataset_name
        assert (self.dataset_name in self.valid_dataset)
        if self.dataset_name == "coco":
            self.images = get_coco_imdb(root_path, split=split, return_labels=False)
        elif self.dataset_name == "pascal":
            self.images = get_pascal_imdb(root_path, split=split, return_labels=False)
        elif self.dataset_name == "coco_10k":
            self.images = get_coco_subset_imdb(root_path, split=split, return_labels=False, subset="cocostuff10k")
        elif self.dataset_name == "coco_iic_subset_train":
            self.images = get_coco_subset_imdb(root_path, split=split, return_labels=False, subset="iic_subset_train")
        elif self.dataset_name == "imagenet_100":
            self.images = get_imagenet_100_imdb(root_path)
        elif self.dataset_name == "cityscapes":
            self.images = get_cityscapes_imdb(root_path, split=split, return_labels=False)
        elif self.dataset_name == "coco10k&pascal":
            self.images_coco = get_coco_subset_imdb(root_path[0], split=split[0], return_labels=False, subset="cocostuff10k")
            self.images_pascal = get_pascal_imdb(root_path[1], split=split[1], return_labels=False)
            self.images = self.images_coco + self.images_pascal
        self.reshuffle()

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        view_1, view_2 = self.transform_image(index, image)
        return index, view_1, view_2

    def reshuffle(self):
        """
        Generate random floats for all image data to deterministically random transform.
        This is to use random sampling but have the same samples during clustering and 
        training within the same epoch. 
        """
        self.shuffled_indices = np.arange(len(self.images))
        np.random.shuffle(self.shuffled_indices)
        self.init_transforms()

    def transform_image(self, index, image):
        # Base transform
        image = self.transform_base(image, Image.BILINEAR)
        # Invariance transform. 
        image1 = self.transform_inv(index, image, 0)
        image1 = TF.resize(image1, self.res1, Image.BILINEAR)
        image1 = self.transform_tensor(image1)
        image2 = self.transform_inv(index, image, 1)
        image2 = TF.resize(image2, self.res1, Image.BILINEAR)
        image2 = self.transform_tensor(image2)
        return image1, image2

    def transform_inv(self, index, image, ver):
        """
        Hyperparameters same as MoCo v2. 
        (https://github.com/facebookresearch/moco/blob/master/main_moco.py)
        """
        if 'brightness' in self.inv_list:
            image = self.random_color_brightness[ver](index, image)
        if 'contrast' in self.inv_list:
            image = self.random_color_contrast[ver](index, image)
        if 'saturation' in self.inv_list:
            image = self.random_color_saturation[ver](index, image)
        if 'hue' in self.inv_list:
            image = self.random_color_hue[ver](index, image)
        if 'gray' in self.inv_list:
            image = self.random_gray_scale[ver](index, image)
        if 'blur' in self.inv_list:
            image = self.random_gaussian_blur[ver](index, image)

        return image

    def transform_eqv(self, indice, image, mode):  # mode
        if 'random_crop' in self.eqv_list:
            image = self.random_resized_crop(indice, image, mode)  # mode
        if 'h_flip' in self.eqv_list:
            image = self.random_horizontal_flip(indice, image)
        if 'v_flip' in self.eqv_list:
            image = self.random_vertical_flip(indice, image)

        return image

    def init_transforms(self):
        N = len(self.images)

        # Base transform.
        self.transform_base = RandomResizedCrop2(self.res2, scale=self.scale)#BaseTransform(self.res2)

        # Transforms for invariance. 
        # Color jitter (4), gray scale, blur. 
        self.random_color_brightness = [RandomColorBrightness(x=0.3, p=0.8, N=N) for _ in range(2)]
        self.random_color_contrast = [RandomColorContrast(x=0.3, p=0.8, N=N) for _ in range(2)]
        self.random_color_saturation = [RandomColorSaturation(x=0.3, p=0.8, N=N) for _ in range(2)]
        self.random_color_hue = [RandomColorHue(x=0.1, p=0.8, N=N) for _ in range(2)]
        self.random_gray_scale = [RandomGrayScale(p=0.2, N=N) for _ in range(2)]
        self.random_gaussian_blur = [RandomGaussianBlur(sigma=[.1, 2.], p=0.5, N=N) for _ in range(2)]

        self.random_horizontal_flip = RandomHorizontalTensorFlip(N=N)
        self.random_vertical_flip = RandomVerticalFlip(N=N)
        self.random_resized_crop = RandomResizedCrop(N=N, res=self.res1, scale=self.scale)

        # Tensor transform. 
        self.transform_tensor = TensorTransform()

    def __len__(self):
        return len(self.images)


class EvalPascal(data.Dataset):
    # Used for evaluation process
    # Adapted from https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation/blob/main/segmentation/data/dataloaders/pascal_voc.py
    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, root_path, split='trainaug', transform=None):
        # Transform
        self.transform = transform
        # Splits are pre-cut
        print("Initializing dataloader for PASCAL VOC12 {} set".format(''.join(split)))
        self.images, self.labels = get_pascal_imdb(root_path, split, return_labels=True)
        assert (len(self.images) == len(self.labels))
        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        # Load image
        img = np.array(Image.open(self.images[index]).convert('RGB'))
        # Load semantic labels
        label = np.array(Image.open(self.labels[index]))
        if label.shape != img.shape[:2]:
            label = cv2.resize(label, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            img, label = self.transform(img, label)
        return img, label

    def __len__(self):
        return len(self.images)

    def get_img_size(self, idx=0):
        img = Image.open(self.images[idx])
        return list(reversed(img.size))

    def __str__(self):
        return 'VOC12(split=' + str(self.split) + ')'

    def get_class_names(self):
        return self.VOC_CATEGORY_NAMES


class EvalCoco(data.Dataset):
    def __init__(self, root_path, split='val', transform=None, coarse_labels=True,
                 data_set="full", subset="iic_subset_train"):
        # Set paths
        valid_splits = ['train', 'val']
        self.root_path = root_path
        self.coarse_labels = coarse_labels
        self.COCO_CATEGORY_NAMES = self.load_class_names()
        # Transform
        self.transform = transform
        self.split = split
        self.data_set = data_set
        self.thing_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # For coarse labels
        self.stuff_idx = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        assert (self.split in valid_splits)
        print("Initializing dataloader for COCO {} set".format(''.join(self.split)))
        print("Loading "+subset+" data")
        self.images, self.labels = get_coco_subset_imdb(root_path, self.split, return_labels=True, subset=subset)
        #self.images, self.labels = get_coco_imdb(root_path, self.split, return_labels=True)
        assert (len(self.images) == len(self.labels))
        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))
        self.fine_to_coarse = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                               13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                               25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                               37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                               49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                               61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                               73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                               85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                               97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                               107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                               117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                               127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                               137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                               147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                               157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                               167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                               177: 26, 178: 26, 179: 19, 180: 19, 181: 24}  # adopted from stego data.py

    def __getitem__(self, index):
        # Load image
        img = np.array(Image.open(self.images[index]).convert('RGB'))
        # Load semantic labels
        label = np.array(Image.open(self.labels[index]))
        if label.shape != img.shape[:2]:
            label = cv2.resize(label, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        if self.transform is not None:
            img, label = self.transform(img, label)
        if self.coarse_labels:
            new_label = torch.zeros_like(label)
            for fine, coarse in self.fine_to_coarse.items():
                if self.data_set == "full":
                    new_label[label == fine] = coarse
                elif self.data_set == "thing":
                    if coarse in self.stuff_idx:  # ignore stuff pixels for thing dataset
                        new_label[label == fine] = 255
                    else:
                        new_label[label == fine] = coarse
                elif self.data_set == "stuff":
                    if coarse in self.thing_idx:  # ignore thing pixels for stuff dataset
                        new_label[label == fine] = 255
                    else:
                        new_label[label == fine] = coarse - 12
                        # stuff label starts from 12, in ce loss, label should start from 0
            new_label[label == 255] = 255
            label = new_label
        return img, label

    def __len__(self):
        return len(self.images)

    def get_img_size(self, idx=0):
        img = Image.open(self.images[idx])
        return list(reversed(img.size))

    def __str__(self):
        return 'COCO(split=' + str(self.split) + ')'

    def get_class_names(self):
        if self.data_set == "stuff":
            return self.COCO_CATEGORY_NAMES[12:]
        elif self.data_set == "thing":
            return self.COCO_CATEGORY_NAMES[0:12]
        return self.COCO_CATEGORY_NAMES

    def load_class_names(self):
        if self.coarse_labels:
            label_class_file_path = self.root_path + "coarse_label.txt"
        else:
            label_class_file_path = self.root_path + "label.txt"
        f = open(label_class_file_path)
        class_list = []
        for line in f:
            class_name = line.split(":")[1].strip()
            if class_name != "unlabeled":  # When label.txt is loaded, "unlabeled" should be ignored.
                class_list.append(class_name)
        return class_list


def get_pascal_imdb(root_path, split, return_labels=True):
    images = []
    labels = []
    valid_splits = ['train_aug', 'train', 'val']
    assert (split in valid_splits)
    print("Loading PascalVoc data")
    split_file = root_path + 'ImageSets/Segmentation/' + split + '.txt'
    with open(split_file, "r") as f:
        lines = f.read().splitlines()
    for line in lines:
        # Images
        image = os.path.join(root_path, line.split()[0])
        assert os.path.isfile(image)
        images.append(image)
        # Labels
        label = os.path.join(root_path, line.split()[1])
        assert os.path.isfile(label)
        labels.append(label)
    assert (len(images) == len(labels))
    if return_labels:
        return images, labels
    return images


def get_coco_imdb(root_path, split, return_labels=True):
    images = []
    labels = []
    valid_splits = ['train', 'val']
    assert (split in valid_splits)
    print("Loading COCO data")
    img_path = root_path + split + "2017"
    label_path = root_path + "stuffthingmaps_trainval2017/" + split + "2017"
    # load file paths for the training set.
    for (root, dirs, files) in os.walk(img_path, topdown=True):
        for file in files:
            images.append(img_path + "/" + file)

    for (root, dirs, files) in os.walk(label_path, topdown=True):
        for file in files:
            labels.append(label_path + "/" + file)
    images.sort()
    labels.sort()
    if return_labels:
        return images, labels
    return images


def get_coco_subset_imdb(root_path, split, return_labels=True, subset="cocostuff10k"):
    images = []
    labels = []
    valid_splits = ['val','train']
    valid_subsets = ['cocostuff10k', 'iic_subset_train', 'iic_subset_val']
    assert (split in valid_splits)
    assert (subset in valid_subsets)
    if subset == "cocostuff10k":
        split_file = root_path + 'cocostuff10k.txt'
    elif subset == "iic_subset_train":
        split_file = root_path + 'Coco164kFull_Stuff_Coarse.txt'
    elif subset == "iic_subset_val":
        split_file = root_path + 'Coco164kFull_Stuff_Coarse_7.txt'
    image_path = root_path + split + "2017/"
    label_path = root_path + "stuffthingmaps_trainval2017/" + split + "2017/"
    with open(split_file, "r") as f:
        lines = f.read().splitlines()
    for line in lines:
        if subset == "cocostuff10k":
            name = line.split("_")[-1]
        else:
            name =line
        # Images
        image = os.path.join(image_path,  name + ".jpg")
        assert os.path.isfile(image)
        images.append(image)
        # Labels
        label = os.path.join(label_path, name + ".png")
        assert os.path.isfile(label)
        labels.append(label)
    images.sort()
    labels.sort()
    if return_labels:
        return images, labels
    return images

