'''
time: 2022.08.24
litianlin
该库包含：从一张经过目标检测的截图中，检测位于中心位置的椭圆的算法
'''
import argparse
import math
import array
import os

import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import ruptures as rpt

def inverse_color(image):
    '''
    对灰度图进行灰度的反转
    '''
    height,width = image.shape
    img2 = image.copy()

    for i in range(height):
        for j in range(width):
            img2[i,j] = (255-image[i,j])
    return img2

def delete_smallarea(img, ratio):
    '''
    删除图片中的雄安面积连通域
    img: 图片array
    ratio: 想要删除的连通区域占图片大小的比例阈值
    '''
    [height, weidth] = img.shape
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)  # contours返回的是轮廓像素点列表，一张图像有几个目标区域就有几个列表值
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])  # 计算轮廓面积，但是可能和像素点区域不太一样 收藏
        if area < height * weidth * ratio:
            cv2.drawContours(img, [contours[i]], 0, 0, -1)  ##去除小面积连通域
    return img

def smooth(y, box_pts):
    '''
    对向量y中的值，进行平滑，方法：滑动窗口卷积
    :param y: 待平滑向量
    :param box_pts: 窗口的长度
    :return: 平滑后的向量
    '''
    num = len(y)
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    jieduan = int(box_pts / 2)
    y[jieduan:num - jieduan - 1] = y_smooth[jieduan:num - jieduan - 1] # 截掉两端的点
    return y

def cal_direction(point_a, point_b, point_c): # 算法导论p596
    '''
    从曲率角度判断点b是否为凹点，即判断向量ac是由向量ab顺时针还是逆时针旋转得到。
    返回1，ac由ab逆时针旋转得到，即从曲率角度点b是凹点；返回-1，不是凹点
    :param point_a: 前点
    :param point_b: 本点
    :param point_c: 后点
    '''
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]  # 点a、b、c的x坐标
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]  # 点a、b、c的y坐标

    if len(point_a) == len(point_b) == len(point_c) == 3:
        a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]  # 点a、b、c的z坐标
    else:
        a_z, b_z, c_z = 0,0,0  # 坐标点为2维坐标形式，z 坐标默认值设为0

    # 向量 m=(x1,y1,z1), n=(x2,y2,z2)
    x1,y1,z1 = (b_x-a_x),(b_y-a_y),(b_z-a_z)
    x2,y2,z2 = (c_x-a_x),(c_y-a_y),(c_z-a_z)

    # 计算叉积，（p2-p0）×（p1-p0）=x2*y1-x1*y2 为负，p0p2在p0p1逆时针方向
    if (x2*y1-x1*y2) >= 0: # 不是凹点
        B = -1
    else: #是凹点
        B = 1
    return B

def iscurve(contour, knee_point ,step):
    '''
    从曲率角度判断knee_point是否是凹点
    :param contour: 曲线
    :param knee_point: 曲线上待判断点的位置
    :param step: 前点、后点到knee_point的距离
    :return: 1，是凹点；-1，不是凹点
    '''
    if knee_point-step<0 or knee_point+step >= contour.shape[0]:
        return -1
    front = contour[knee_point-step]
    now = contour[knee_point]
    behind = contour[knee_point+step]
    flag = cal_direction(front, now, behind)
    return flag

def ellipse_detect(lr_img, times):
    '''
    从给定的小尺寸图片中，检测出一个位于最中心的椭圆
    input: lr_img 待检测图片array格式
    output: ellipse 检测到的位于最中心的椭圆，其椭圆的中心坐标，长短轴长度（2a，2b），旋转角度
            area 椭圆的面积
    '''

    # 基于三次插值的图像重建
    # cv2.imwrite('/home/guangpu120/ltl/yolov5/data/images/lr_img.jpg', lr_img)  # debug用，保存了最后一张crop
    hr_img = cv2.resize(lr_img, (0, 0), fx=times, fy=times, interpolation=cv2.INTER_CUBIC)
    [height, weidth, deep]= hr_img.shape

    # 局部阈值（动态阈值，平均法）
    gray = cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)  # 自适应阈值,平均法
    binary = cv2.bitwise_not(binary) # 二值化反转

    # 开运算 先腐蚀后膨胀,去除毛刺和小粘连
    kernel = np.ones((3, 3), dtype=np.uint8)
    open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, 1)

    # 删除连通域面积小于原图10%的部分
    del_small = delete_smallarea(open, 0.1)

    # 图像填充,首先填充闭合轮廓
    contours, hierarchy = cv2.findContours(del_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # cv2.RETR_EXTERNAL是只检测外轮廓
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(del_small, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)
    hole_fill = sum(contour_list)

    # 图像填充，再将图像反色，去除小面积的连通域，再反色
    if (type(hole_fill) == int): # 如果yolo检测到的图片中没有气泡
        return (0,0,0,0,0),0,0
    hole_fill = inverse_color(hole_fill)
    hole_fill = delete_smallarea(hole_fill, 0.1)
    hole_fill = inverse_color(hole_fill)

    # 基于凹凸变化的凹点检测
    # 获取每段轮廓的质心,并根据每段轮廓到图片中心的距离，选择出在图片中央的轮廓，认为是这一张crop要检测的那个气泡
    contours, hierarchy = cv2.findContours(hole_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # cv2.RETR_EXTERNAL是只检测外轮廓
    len_contour = len(contours)
    standard = 100000 # 判断图片最中心需要识别的区域，边缘的认为是在其他图片中可以检测到的
    cent_x=[]
    cent_y=[]
    for i in range(len_contour): # contours是先x后y,逆时针排列
        # mass_x and mass_y are the list of x indices and y indices of mass pixels
        cent_x.append(np.average(contours[i][:,0,0]))
        cent_y.append(np.average(contours[i][:,0,1]))# 计算每段轮廓的质心位置
        distance_centroid_piccenter = pow(pow(cent_x[i]-weidth/2,2)+pow(cent_y[i]-height/2,2),0.5) # 每段轮廓的质心到图片中心的距离
        if standard > distance_centroid_piccenter:
            a = i
            standard = distance_centroid_piccenter
        cent_point = (int(cent_x[a])), (int(cent_y[a]))  # 注意是先x再y
        contour = np.reshape(contours[a], (-1, 2))
        center = hole_fill.copy()
        center[cent_point[1],cent_point[0]] = 0 # 只有mat类型变量访问时，下标是反着写的，（y，x）其他都是(x,y)

    # 计算边界点到质心的距离,并绘制图像
    distance=[]
    num = contour.shape[0]
    for i in range(num):
        y = np.array(cent_point[1])
        distance.append(pow(pow(contour[i,0]-cent_point[0],2)+pow(contour[i,1]-cent_point[1],2),0.5))

    # 极小值点检测
    valley = []
    points = np.array(distance) # points是距离坐标图上的点

    # 用滑动窗口卷积来平滑
    win_len = 3
    points = smooth(points,win_len)

    # kenn函数检测拐点
    algo = rpt.Pelt(model="clinear", jump=1).fit(points)
    result = algo.predict(pen=10)
    knee = result[0:len(result)-1] # 删除掉最后一个点

    # 曲率验证
    flag =[]
    for i in range(len(knee)):
        is_curve = iscurve(contour, knee[i], 5)
        flag.append(is_curve)
        if is_curve == 1:
            valley.append(knee[i])

    # print(valley)   # valley是极小值点的位置
    len_valley = len(valley)
    valley_selected = []
    img_extreme_point = hole_fill.copy() # 绘制检测到的极小值点
    for i in range(len_valley):  # 删除位于图片四条边上的极值点
        x = (contour[valley[i]])[0]
        y = (contour[valley[i]])[1]
        if ~((x==0 or x==(weidth - 1)) or (y==0 or y==(height - 1))):
            valley_selected.append(valley[i])

    # 分割路径；contour里面的点是从最上面的点沿顺时针排列的
    # 再次判断，几段轮廓中，哪段在图像中央
    standard = 10000
    len_valley_selected = len(valley_selected)
    curve_test = []
    d = []
    curve = []
    if len_valley_selected == 0:
        curve = contour
    else:
        for i in range(len_valley_selected):
            if i == 0:
                curve_test.append(np.vstack((contour[valley_selected[len(valley_selected)-1]:],contour[0:valley_selected[0]]))) # 当为第一段曲线时，用最后一段曲线加上开头的一段
            else:
                curve_test.append(contour[valley_selected[i-1]:valley_selected[i]])
            arg_x = np.average(curve_test[i][:,0]) # 计算每段轮廓的平均位置
            arg_y = np.average(curve_test[i][:,1])
            d.append(pow(pow(arg_x - weidth / 2, 2) + pow(arg_y - height / 2, 2),0.5))  # 每段轮廓的质心到图片中心的距离
        d_min = min(d)
        for j in range(len(d)):
            if d[j] - d_min < 0.1 * d_min:  # 如果两段曲线距离图片中心点的距离的差值，比距中心最短的曲线距离的10%小，则认为这些曲线共同构成了中心的气泡
                curve.extend(curve_test[j])

    # 椭圆拟合
    curve = np.array(curve)
    if curve.shape[0] < 5:
        _ellipse = cv2.fitEllipse(contour)  # 椭圆拟合
    else:
        _ellipse = cv2.fitEllipse(curve)  # 椭圆拟合  椭圆的中心坐标，长短轴长度（2a，2b），旋转角度

    ellipse = [_ellipse[0][0]/times,_ellipse[0][1]/times,_ellipse[1][0]/times,_ellipse[1][1]/times,_ellipse[2]]
    area = math.pi * ellipse[2]/2  * ellipse[3]/2  # 椭圆面积计算
    vol = (4/3) * (math.pi) * ellipse[2]/2 * ellipse[3]/2 *(ellipse[2]+ellipse[3])/4 # 估算气泡的体积，纵轴取长短轴的平均值
    return ellipse, area, vol
