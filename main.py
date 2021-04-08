# 用word2vec算法训练词向量，并比较CBOW和Skip-gram训练出来的结果。
# （随机挑选5个词，
# 1、附图展示对于同一个词的用CBOW训练完后相似度最近的10个词和用Skip-gram训练完后相似度最近的10个词，并计算相似度结果；
# 2、输出这5个词的词向量）
# 数据集自己寻找，根据自己电脑配置划分合适的数据集大小。

import collections
import d2lzh as d2l
import math
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import random
import sys
import time
import zipfile



