# -- coding: utf-8 --
import tensorflow as tf
import imgaug.augmenters as iaa
import cv2 as cv
import csv
import os


def augumentor(image):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(
            rotate=(-10, 10)
        ),
        sometimes(iaa.Crop(percent=(0, 0.1), keep_size=True))
    ],
        random_order=True
    )
    image_aug = seq.augment_image(image)
    return image_aug


def aug_gen():
    source_dir = './dataset/data/'
    source_content = './dataset/training.csv'
    auged_images_dir = './dataset/data_auged/'
    auged_content = './dataset/train_aug.csv'

    if not os.path.exists(auged_images_dir):
        os.mkdir(auged_images_dir)

    with open(auged_content, 'w', newline='') as f:
        writer = csv.writer(f)
        with open(source_content, 'r') as f_source:
            reader = csv.reader(f_source)
            for row in reader:
                if row[0] != 'FileID':
                    imagename, label = row[0], int(row[1])
                    image = cv.imread(source_dir + imagename + '.jpg')
                    image_aug = augumentor(image)
                    cv.imwrite(auged_images_dir + imagename + '_aug.jpg', image_aug)
                    writer.writerow([imagename + '_aug', label])


def aug_gen_test():
    source_dir = './dataset/data/'
    source_content_file = './dataset/test.csv'
    auged_images_dir = './dataset/data_auged_test/'
    auged_content_dir = './dataset/test_auged.csv'

    if not os.path.exists(auged_images_dir):
        os.mkdir(auged_images_dir)

    with open(auged_content_dir, 'w', newline='') as l:
        writer = csv.writer(l)
        with open(source_content_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] != 'FileID':
                    imagename = row[0]
                    image = cv.imread(source_dir + imagename + '.jpg')
                    image_aug = augumentor(image)
                    cv.imwrite(auged_images_dir + imagename + '_aug.jpg', image_aug)
                    writer.writerow([imagename + '_aug'])

    with open(auged_content_dir, 'r') as f:
        reader = csv.reader(f)
        # 将数据集存储为TFRecord文件
        with tf.io.TFRecordWriter('./cache/test_aug_tfrecord') as writer:
            for row in reader:
                imagename = row[0]
                image = open(auged_images_dir + imagename + '.jpg', 'rb').read()
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

    raw_dataset = tf.data.TFRecordDataset('./cache/test_aug_tfrecord')

    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_example(example_string):
        feature_dict = tf.io.parse_single_example(example_string, feature_description)
        feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])
        return feature_dict['image']

    test_aug_dataset = raw_dataset.map(_parse_example)

    return test_aug_dataset
