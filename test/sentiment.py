#! -*- coding:utf-8 -*-
# 情感分析例子

import os, json
import numpy as np
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
import jieba
jieba.initialize()

num_classes = 2
maxlen = 512
batch_size = 32

# bert配置
config_path = '/root/kg/bert/chinese_wobert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_wobert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_wobert_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text[:128], int(label)))
    return D


# 加载数据集
train_data = load_data('datasets/sentiment/sentiment.train.data')
valid_data = load_data('datasets/sentiment/sentiment.valid.data')
test_data = load_data('datasets/sentiment/sentiment.test.data')

# 建立分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


bert = build_transformer_model(
    config_path,
    checkpoint_path,
    return_keras_model=False,
)

output = bert.model.output
output = Lambda(lambda x: x[:, 0], name='CLS-token')(output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model_sentiment.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )

    model.load_weights('best_model_1.weights')
    print(u'final test acc: %05f\n' % (evaluate(test_generator)))

else:

    model.load_weights('best_model_sentiment.weights')
