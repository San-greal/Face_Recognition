import cv2
import dlib
import numpy
import math
import os
from os import walk
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
IMAGE_SIZE = 250

#返回默认的人脸检测器
detector = dlib.get_frontal_face_detector()
#该对象是一个工具,它接收包含某个对象的图像区域并输出一组定义该对象姿态的点位置.其中最经典的例子就是人脸姿势预测,它将人脸的图像作为输入,并预计能够识别重要面部标志的位置,例如嘴角和眼睛的角落,鼻尖等向前.
predictor = dlib.shape_predictor(PREDICTOR_PATH)

#获得面部标签点
def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        pass
    if len(rects) == 0:
        pass
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

#在图片上将面部标签点标上
def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
    return im

#将上面两个函数结合起来
def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)
    return im, s

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
                               M=cv2.getRotationMatrix2D(center=(center[0][0], center[0][1]), angle=angle, scale=1.0),
                               dsize=(250, 250))
    output_im2 = cv2.warpPerspective(src=output_im,
                                     M=cv2.getPerspectiveTransform(numpy.array(SrcPoints), numpy.array(CanvasPoints)),
                                     dsize=(250, 250))
    return output_im2

#创建一个文件路径的列表
f = []
for (dirpath, dirnames, filenames) in walk("/home/duzhaoteng/PycharmProjects/opencv/lfw/"):
    for i in dirnames:
        f.append("/home/duzhaoteng/PycharmProjects/opencv/lfw/" + i)

#对我的图片全部进行预处理并且输出储存图像
for i in f:
    for (dirpath, dirnames, filenames) in walk(i):
        for j in filenames:
            try:
                img, landmarks = read_im_and_landmarks(dirpath + "/" + j)
                cv2.imwrite('/home/duzhaoteng/PycharmProjects/opencv/manageface/' + j,
                            annotate_landmarks(resize_image(img, landmarks), landmarks))
            except IndexError:
                pass
        continue





