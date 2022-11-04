# 预计：简化检测过程，实现批量检测
import argparse
import math
import array
import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

def ellipse_detect (lr_img, times):
    '''
    从给定的小尺寸图片中，检测出一个位于最中心的椭圆
    input: lr_img 待检测图片array格式
    output: ellipse 检测到的位于最中心的椭圆，其椭圆的中心坐标，长短轴长度（2a，2b），旋转角度
            area 椭圆的面积
            volume （暂无） 椭圆的体积公式 4/3pi*a*b*c 缺少一个c半径
    '''
    # lr_img = cv2.imread(img_path)

    # 基于三次插值的图像重建
    # times = 2; # 上采样的倍数
    hr_img = cv2.resize(lr_img, (0, 0), fx=times, fy=times, interpolation=cv2.INTER_CUBIC)
    [height, weidth, deep]= hr_img.shape
    cv2.imwrite('/home/guangpu120/ltl/yolov5/data/images/hr_img.jpg',hr_img) # debug用，保存了最后一张crop

    # 局部阈值（动态阈值，平均法）
    gray = cv2.cvtColor(hr_img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 5)  # 自适应阈值,平均法
    binary = cv2.bitwise_not(binary) # 二值化反转
    # cv2.imshow('binary',binary)

    # 开运算 先腐蚀后膨胀,去除毛刺和小粘连
    kernel = np.ones((5, 5), dtype=np.uint8)
    open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, 1)
    # cv2.imshow('open', open)

    # 删除连通域面积小于40的部分
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(open) # stats包含每个连通域外接矩形的左上角坐标x,y;w,h;s(像素个数)
    i=0
    for istat in stats:
        if istat[4]<40:
            if istat[3]>istat[4]:
                r=istat[3]
            else:r=istat[4]
            # cv2.rectangle(open,tuple(istat[0:2]),tuple(istat[0:2]+istat[2:4]) , 0,thickness=-1)  # 26
        i=i+1
    # cv2.imshow('delate_smallpart',open)

    # 图像填充
    contours, hierarchy = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # cv2.RETR_EXTERNAL是只检测外轮廓
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(open, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)
    hole_fill = sum(contour_list)
    # cv2.imshow('hole_fill',hole_fill)

    # 基于凹凸变化的凹点检测
    # 获取每段轮廓的质心,并根据每段轮廓到图片中心的距离，选择出在图片中央的轮廓，认为是这一张crop要检测的那个气泡
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
    # cv2.imshow('center', center)
    # print(cent_point)

    # 计算边界点到质心的距离,并绘制图像
    distance=[]
    num = contour.shape[0]
    for i in range(num):
        y = np.array(cent_point[1])
        distance.append(pow(pow(contour[i,0]-cent_point[0],2)+pow(contour[i,1]-cent_point[1],2),0.5))
    # print(distance)
    # plt.scatter(list(range(num)), distance)
    # plt.show()

    # 极小值点检测
    points = np.array(distance) # points是距离坐标图上的点
    valley, properties = scipy.signal.find_peaks((-points), prominence=(5, None)) # promonence设置凸起的阈值，注意这里用的是反转的 -re
    # print(valley)   # valley是极小值点的位置
    len_valley = len(valley)
    valley_selected = []
    # img_extreme_point = hole_fill.copy() # 绘制检测到的极小值点
    for i in range(len_valley):  # 删除位于图片四条边上的极值点
        x = (contour[valley[i]])[0]
        y = (contour[valley[i]])[1]
        if ~((x==0 or x==(weidth - 1)) or (y==0 or y==(height - 1))):
            valley_selected.append(valley[i])
            # cv2.circle(img_extreme_point,[x, y],15,(0,0,0),-1)
    # cv2.imshow('extreme_point',img_extreme_point)

    # 分割路径；contour里面的点是从最上面的点沿顺时针排列的
    # 再次判断，几段轮廓中，哪段在图像中央
    standard = 10000
    edge_clone = hr_img.copy()
    len_valley_selected = len(valley_selected)
    if len_valley_selected == 0:
        curve = contour
    else:
        for i in range(len_valley_selected):
            curve_test = []
            if i == 0:
                curve_test = np.vstack((contour[valley_selected[len(valley_selected)-1]:],contour[0:valley_selected[0]])) # 当为第一段曲线时，用最后一段曲线加上开头的一段
            else:
                curve_test = contour[valley_selected[i-1]:valley_selected[i]]
            arg_x = np.average(curve_test[:,0]) # 计算每段轮廓的平均位置
            arg_y = np.average(curve_test[:,1])
            d = pow(pow(arg_x - weidth / 2, 2) + pow(arg_y - height / 2, 2),0.5)  # 每段轮廓的质心到图片中心的距离
            if standard > d:
                num = i
                standard = d
                curve = curve_test
    # 椭圆拟合
    if curve.shape[0] < 5:
        _ellipse = cv2.fitEllipse(contour)  # 椭圆拟合
    else:
        _ellipse = cv2.fitEllipse(curve)  # 椭圆拟合
    # print(_ellipse)  # 这儿包含 椭圆的中心坐标，长短轴长度（2a，2b），旋转角度
    area = math.pi * _ellipse[1][0]/2/times * _ellipse[1][1]/2/times # 椭圆面积计算
    return _ellipse, area

#img_path = '/home/guangpu120/ltl/yolov5/runs/detect/exp5/crops/bubbles/012.jpg'
#lr_img = cv2.imread(img_path)
#ellipse_detect(lr_img, 2)
