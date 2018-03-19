import cv2
import numpy
import os
import random
import tensorflow as tf
import tensorlayer as tl
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 

sess = tf.InteractiveSession()


# 读取训练数据
images = []
labels = []


def read_path(path_name):
    for dir_item in os.listdir(path_name):
        # 从初始路径开始叠加，合并成可识别的文件路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            read_path(full_path)
        else:  # 文件
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)
                images.append(image)
                labels.append(path_name)

    return images, labels


# 从指定路径读取训练数据
def load_dataset(path_name):
    images, labels = read_path(path_name)
    images = numpy.array(images)
    print(images.shape)

    # 标注数据，'A'文件夹下都是A的脸部图像，全部指定为0，另外一个文件夹下是B的，全部指定为1
    labels = numpy.array([0 if label.endswith('A') else 1 for label in labels])

    return images, labels


images, labels = load_dataset("/home/duzhaoteng/PycharmProjects/opencv/data/")
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.5,random_state=random.randint(0, 100))
train_images, valid_images, train_labels, valid_label = train_test_split(train_images, train_labels, test_size=0.3,random_state=random.randint(0, 100))


train_images = train_images.astype('float32')            
valid_images = valid_images.astype('float32')
test_images = test_images.astype('float32')

# define placeholder
x = tf.placeholder(tf.float32, shape=[16,250,250,3], name='x')
y_ = tf.placeholder(tf.int64, shape=[None,], name='y_')

#设置网络模型
with tf.variable_scope('1'):
    network = tl.layers.InputLayer(x, name='input')
    network = tl.layers.Conv2dLayer(network,shape=[5,5,3,32],padding ='SAME',strides = [1,1,1,1],name='conv1')
    network = tl.layers.MaxPool2d(network,filter_size = (5,5),padding ='SAME',name='maxpool1')
    network = tl.layers.Conv2dLayer(network, shape=[5,5,32,64], padding='SAME', strides=[1, 1, 1, 1], name='conv2')
    network = tl.layers.MaxPool2d(network, filter_size=(5, 5), padding='SAME', name='maxpool2')
    network = tl.layers.FlattenLayer(network,name='flatten')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
    network = tl.layers.DenseLayer(network, n_units=10, act=tf.nn.relu, name='relu1')
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    network = tl.layers.DenseLayer(network, n_units=2, act=tf.identity, name='output')

# 设置cost函数
y = network.outputs
cost = tl.cost.cross_entropy(y, y_, name='cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# 设置一个优化函数
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)

# 在session里面初始化所有的变量
tl.layers.initialize_global_variables(sess)

# 输出网络的信息
network.print_params()
network.print_layers()

# 训练网络
tl.utils.fit(
    sess, network, train_op, cost, train_images, train_labels, x, y_, acc=acc, batch_size=16, n_epoch=100, print_freq=10, X_val=valid_images, y_val=valid_label, eval_train=False)

# 测试
tl.utils.test(sess, network, acc, test_images, test_labels, x, y_, batch_size=16, cost=cost)

# 保存结果
tl.files.save_npz(network.all_params, name='model.npz')
sess.close()

