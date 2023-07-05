#!/usr/bin/env python
import numpy as np
import cupy as cp
import cv2
import xml.dom.minidom

import logging
from logging.handlers import RotatingFileHandler
import os
import time
import cucim
    
def get_logger(save_path, logger_name):
    """
    Initialize logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    # file log
    file_handler = RotatingFileHandler(save_path, maxBytes=2*1024*1024, backupCount=1)
    file_handler.setFormatter(file_formatter)

    # console log
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    # logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

DEFAULT_LOGGER = get_logger(os.getcwd() + 'log', __name__)


def load_xml(file_path):
    """读取xml文件，
    返回总坐标列表xy_list:存放一张图片上所有画出域的点 """

    # 用于打开一个xml文件，并将这个文件对象dom变量
    dom = xml.dom.minidom.parse(file_path)
    # 对于知道元素名字的子元素，可以使用getElementsByTagName方法获取
    annotations = dom.getElementsByTagName('Annotation')

    # 存放所有的 Annotation
    xyi_in_annotations = []

    for Annotation in annotations:
        # 存放一个 Annotation 中所有的 X,Y值
        xy_in_annotation = []
        # 读取 Coordinates 下的 X Y 的值
        coordinates = Annotation.getElementsByTagName("Coordinate")
        for Coordinate in coordinates:
            list_in_annotation = []
            x = int(float(Coordinate.getAttribute("X")))
            y = int(float(Coordinate.getAttribute("Y")))
            list_in_annotation.append(x)
            list_in_annotation.append(y)
            xy_in_annotation.append(list_in_annotation)
        xyi_in_annotations.append(xy_in_annotation)
    return xyi_in_annotations

def judge_position(contours, point_list):
    """去除轮廓以外选中的干扰区域
    :param contours: 轮廓
    :param point_list: 滑窗的顶点和中心坐标列表
    :return: 是否在轮廓中的状态列表 1在轮廓中 -1 不在轮廓中
    """
    out = 0
    for point in point_list:
        point_statu = cv2.pointPolygonTest(contours, point, False)
        if out == 0 and point_list == -1:
            return -1
        if point_statu == 1:
            out = out + 1
    return out

def get_area_ratio(img):
    """去除轮廓内的干扰区域
    :param img: 滑动窗口
    :return: 面积比
    """
    if isinstance(img, cp.ndarray):
        img = img.get()
    img = np.array(img)

    img = cv2.GaussianBlur(img, (3, 3), 0)#高斯滤波去噪
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#颜色空间的转换
    _, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)#二值化处理，大于200变255，小于200变0，必须是单通道

    # 得到轮廓
    contous, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#检索轮廓，输入为二值化图像，第一个参数只检测外轮廓。
    blank_area_size = 0
    for contou in contous:
        area = cv2.contourArea(contou)
        blank_area_size = blank_area_size + area

    img_w = img.shape[0]
    img_h = img.shape[1]
    area_ratio = blank_area_size/(img_h * img_w)
    return area_ratio

def cost_time(func):
    # https://blog.csdn.net/weixin_38924500/article/details/111679503
    def fun(*args, **kwargs):
        t = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s') # with {args} and {kwargs} 
        return result

    return fun
    
@cost_time
def cu_image_to_array(img):
    '''
    img: result from CuImage.read_region
    returns: cp.array if img is on cuda; np.array if img is on cpu.
    '''
    if isinstance(img, cucim.CuImage):
        if 'cuda' in str(img.device):
            if str(img.device) == 'cuda':
                img = cp.array(img)
            else:
                with cp.cuda.Device(int(str(img.device).replace('cuda:',''))):
                    img = cp.array(img)
        elif str(img.device) == 'cpu':
            img = np.array(img)
        else:
            raise Exception('device of CuImage should be cpu or cuda or cuda:number, got {}'.format(str(img.device)))
    return img