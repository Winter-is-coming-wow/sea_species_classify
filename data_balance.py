# -- coding: utf-8 --
import tensorflow as tf
import imgaug.augmenters as iaa
import csv
import cv2 as cv


def augumentor(image):
    seq = iaa.Sequential(
        [
            iaa.Affine(  # 对一部分图像做仿射变换
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 图像缩放为80%到120%之间
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移±20%之间
                order=[0, 1]  # 使用最邻近差值或者双线性差值
            ),
            iaa.Crop(percent=(0, 0.1), keep_size=True)
        ],
        random_order=False
    )
    image_aug = seq.augment_image(image)

    return image_aug


def balance():
    source_dir = './dataset/data/'
    species_num = [i - i for i in range(20)]

    with open('./dataset/training.csv', 'r') as train_f:
        reader = csv.reader(train_f)
        for row in reader:
            if row[0] != 'FileID':
                label = int(row[1])
                species_num[label] += 1
    max_num = max(species_num)

    train_f.close()

    Add = []
    with open('./dataset/training.csv', 'r') as f:
        f.seek(0, 0)
        reader = csv.reader(f)
        for row in reader:
            if row[0] != 'FileID':
                filename = row[0]
                label = int(row[1])
                index = species_num[label]
                if index < max_num:
                    image = cv.imread(source_dir + filename + '.jpg')
                    auged_image = augumentor(image)
                    cv.imwrite(source_dir + filename + str(index) + '.jpg', auged_image)
                    Add.append([filename + str(index), label])
                    species_num[label] += 1
    with open('./dataset/training.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        for row in Add:
            writer.writerow(row)
