## 简单的人脸识别程序

### 包含文件：

> manageface.py
>
> facerecognize.py
>
> lfw为训练图像文件
>
> data为图像预处理后从中选取出进行识别的两个人的人脸图像
>
> model.npz为运行程序后进行记录的文件

#### 1.将图像进行预处理

#对图像进行预处理并且重新得到预处理的函数

    
    def resize_image(img,landmarks):
    	for i, d in enumerate(detector(img, 1)):
        	x1 = d.top() if d.top() > 0 else 0
        	y1 = d.bottom() if d.bottom() > 0 else 0
        	x2 = d.left() if d.left() > 0 else 0
        	y2 = d.right() if d.right() > 0 else 0
        	SrcPoints = numpy.float32([[x2, x1],
                                   		[y2, x1],
                                   		[x2, y1],
                                   		[y2, y1]])
        	CanvasPoints = numpy.float32([[0, 0], [250, 0], [0, 250], [250, 250]])
    	center = numpy.asarray((landmarks[37] + landmarks[44]) * 0.5)
    	angle = numpy.arctan2(numpy.asarray(landmarks[44] - landmarks[37])[0][1],
                          	numpy.asarray(landmarks[44] - landmarks[37])[0][0]) * 180 / math.pi
    	output_im = cv2.warpAffine(src=img,
                               	M=cv2.getRotationMatrix2D(center=(center[0][0], 		center[0][1]), angle=angle, scale=1.0),
        	                    dsize=(250, 250))
    	output_im2 = cv2.warpPerspective(src=output_im,
                                     			M=cv2.getPerspectiveTransform(numpy.array(SrcPoints), numpy.array(CanvasPoints)),
                                     dsize=(250, 250))
    return output_im2
这部分代码之中通过眼睛位置从而进行图像旋转后将图像的双眼与图像边沿平行，有利于训练的准确度提升（眼睛的位置则是通过dlib面部标签点识别检测出来的）。另外本代码还将图像进行归一化，灰度处理从而使其光照的问题得到一定解决。（选取面部最大特征）

#### 2.编写一个基本的CNN

#设置网络模型

    with tf.variable_scope('1'):	
    	network = tl.layers.InputLayer(x, name='input')
    	network = tl.layers.Conv2dLayer(network,shape=[5,5,3,32],padding 					='SAME',strides = [1,1,1,1],name='conv1')
    	network = tl.layers.MaxPool2d(network,filter_size = (5,5),padding 					='SAME',name='maxpool1')
    	network = tl.layers.Conv2dLayer(network, shape=[5,5,32,64], padding='SAME', 		strides=[1, 1, 1, 1], name='conv2')
    	network = tl.layers.MaxPool2d(network, filter_size=(5, 5), padding='SAME', 			name='maxpool2')
    	network = tl.layers.FlattenLayer(network,name='flatten')
    	network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')
    	network = tl.layers.DenseLayer(network, n_units=10, act=tf.nn.relu, 				name='relu1')
    	network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
    	network = tl.layers.DenseLayer(network, n_units=2, act=tf.identity, 				name='output')
这个是一个基本的两层卷积神经网络，运用了卷积和池化的基本操作，最后用最基本的“RELU”函数进行激活。

####3.训练预测

由于该网络写的不深，也没有运用一些已有的高精度模型（FaceNet到时候复现一下好了），val acc基本保持在75-80%左右，也可能是因为选取的两人照片太少，只有二十张左右（data数据集中A与B两人）。