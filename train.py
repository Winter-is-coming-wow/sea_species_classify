# -- coding: utf-8 --
import tensorflow as tf
import os
from gl import *

def training(train_dataset, model, epoch, step, model_type):
    train_data = train_dataset.map(lambda image, label: (tf.image.resize(image, input_size) / 255.0, label),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_data = train_data.shuffle(buffer_size=1000)
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    history = model.fit(train_data, epochs=epoch)
    print(history.history.keys())
    model.save_weights('./dataset/model_weights/' + model_type + step + '.h5')


def finetune(train_dataset, model_type, model):
    print('====================开始训练==================')
    if not os.path.exists('./dataset/model_weights/'):
        os.mkdir('./dataset/model_weights/')

    if model_type == 'efficientnet':
        epoch = [60, 60, 60, 60, 60, 60]
        layers_name = ['block' + str(i) + 'a_expand_conv' for i in reversed(range(7)) if i > 1]

        print('====================微调全连接层==================')
        pretrain_net = model.layers[0]
        pretrain_net.trainable = False
        training(train_dataset, model, epoch[0], '1', model_type)

        for i in range(6):
            if i < 5:
                print('===================微调第' + str(6 - i) + '块及以后=================')
                model.load_weights('./dataset/model_weights/' + model_type + str(i + 1) + '.h5')
                pretrain_net = model.layers[0]
                pretrain_net.trainable = True
                set_trainable = False
                for layer in pretrain_net.layers:
                    if layer.name == layers_name[i]:
                        set_trainable = True
                    if set_trainable:
                        layer.trainable = True
                    else:
                        layer.trainable = False
                training(train_dataset, model, epoch[i + 1], str(i + 2), model_type)

        model.save('cache/my_model.h5')
