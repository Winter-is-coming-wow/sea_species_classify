# -- coding: utf-8 --
import tensorflow as tf
from gl import *
from data_aug import aug_gen_test


def predict(test_dataset, model):
    test_data = test_dataset.map(lambda image, label: (tf.image.resize(image, input_size) / 255.0),
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_data1 = test_dataset.map(lambda image, label: (tf.image.resize(image, input_size) / 255.0, label),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_data1 = test_data1.batch(batch_size)
    test_y = []
    for i in test_dataset:
        test_y.append(i[1].numpy())
    print(test_y)
    test_data = test_data.batch(batch_size)

    print('====================开始预测==================')
    y_pre = model.predict(test_data)
    print(model.evaluate(test_data1))

    test_aug_dataset = aug_gen_test()
    test_aug_data = test_aug_dataset.map(lambda image: (tf.image.resize(image, [224, 224]) / 255.0),
                                         num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_aug_data = test_aug_data.batch(batch_size)
    y_aug_pre = model.predict(test_aug_data)
    Y_pre = (y_pre + y_aug_pre) / 2
    Y_pre = tf.argmax(Y_pre, axis=1)

    check = [Y_pre[i].numpy() == test_y[i] for i in range(len(test_y))]
    precision = sum(check) / len(check)
    print(precision)
    print('====================预测结束==================')
