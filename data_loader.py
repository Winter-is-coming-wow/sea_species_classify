# -- coding: utf-8 --
import tensorflow as tf
import csv
import os

def load_data():
    # 构建训练集
    sourcedir = './dataset/data/'
    augdir = './dataset/data_auged/'
    if not os.path.exists('./cache'):
        os.mkdir('./cache')
    with tf.io.TFRecordWriter('./cache/train_tfrecord') as writer:
        with open('./dataset/training.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] != 'FileID':
                    filename, label = row[0], int(row[1])
                    image = open(sourcedir + filename + '.jpg', 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())

        with open('./dataset/train_aug.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                filename, label = row[0], int(row[1])
                image = open(augdir + filename + '.jpg', 'rb').read()
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())

        raw_dataset = tf.data.TFRecordDataset('./cache/train_tfrecord')
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)
        }

        def _parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, feature_description)
            feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'])
            return feature_dict['image'], feature_dict['label']

        train_dataset = raw_dataset.map(_parse_example)

    # 构建测试集
    with tf.io.TFRecordWriter('./cache/test_tfrecord') as writer:
        with open('./dataset/annotation.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] != 'FileID':
                    imagename, label = row[0], int(row[1])
                    image = open(sourcedir + imagename + '.jpg', 'rb').read()
                    feature = {
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),

                        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
    raw_dataset = tf.data.TFRecordDataset('./cache/test_tfrecord')
    test_dataset = raw_dataset.map(_parse_example)

    return train_dataset, test_dataset
