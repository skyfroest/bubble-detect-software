import time

from PyQt5.QtCore import QTimer

from detect_bubbles import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from detect import detect_img

import cv2
import torch
import numpy as np
import os

from pathlib import Path

from utils.general import increment_path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

class UiMain(QMainWindow, Ui_MainWindow):
    # 初始化，设置槽函数
    def __init__(self, parent=None):
        super(UiMain, self).__init__(parent)
        self.setupUi(self)
        # 打开
        self.img.triggered.connect(self.loadImage)
        self.dir.triggered.connect(self.loadDir)
        self.my_timer = QTimer() # 初始化定时器
        self.my_timer.timeout.connect(self.my_timer_cb) # 创建定时器任务
        self.camera.triggered.connect(self.loadCamera)
        # 保存
        self.save_dir.triggered.connect(self.saveDir)
        # 运行
        self.run_all.triggered.connect(self.detect)
        self.runButton.clicked.connect(self.detect)
        # 空白参数
        self.save_dir_name = ''
        self.fordetect = ''
        self.btn_status = False;
        self.setFixedSize(self.size());


    # 打开单张图片
    def loadImage(self):
        self.fname, _ = QFileDialog.getOpenFileName(self, '请选择图片','.','图像文件(*.jpg *.jpeg *.png)')
        if self.fname:
            self.run_info.setText("文件打开成功 "+self.fname)
            img = QtGui.QPixmap(self.fname).scaled(self.origin_img.width(), self.origin_img.height())
            self.origin_img.setPixmap(img)
            self.fordetect = self.fname
        else:
            self.run_info.setText("打开文件失败")

    # 打开文件夹
    def loadDir(self):
        self.dirname = QFileDialog.getExistingDirectory(self, '请选择文件夹', './')
        if self.dirname:
            self.run_info.setText("文件夹打开成功 "+self.dirname)
            files = os.listdir(self.dirname)
            fnames = []
            for file in files:
                # os.path.splitext():分离文件名与扩展名
                suffix = os.path.splitext(file)[1]
                if suffix == '.jpg' or suffix == '.jpeg' or suffix == '.png':
                    fnames.append(file)
            if fnames:
                self.fname = self.dirname +'/' + fnames[len(fnames)-1]
                img = QtGui.QPixmap(self.fname).scaled(self.origin_img.width(), self.origin_img.height())
                self.origin_img.setPixmap(img)
                self.fordetect = self.dirname
            else:
                self.run_info.setText("文件夹中不包含图片文件")

        else:
            self.run_info.setText("打开文件失败")

    # 打开摄像头
    def loadCamera(self):
        self.fordetect = '1'
        self.my_timer.start(1000)  # 25fps
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # 定时器任务
    def my_timer_cb(self):
        if self.cap:
            """图像获取"""
            ret, self.image = self.cap.read()
            show = cv2.resize(self.image, (512, 512))
            show = cv2.flip(show, 1)
            self.show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            """结果呈现"""
            showImage = QImage(self.show.data, self.show.shape[1], self.show.shape[0], QImage.Format_RGB888)
            self.origin_img.setPixmap(QPixmap.fromImage(showImage))

    # 开始检测
    def detect(self):
        # 创建保存文件夹
        if (self.save_dir_name):
            self.save_dir = increment_path(Path(self.save_dir_name) / 'exp', exist_ok=False)  # increment run
        else:
            self.save_dir = increment_path(Path(ROOT / 'runs') / 'exp', exist_ok=False)
        (self.save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make dir

        # 检测输入数据类型
        if not(self.fordetect):
            self.run_info.setText("请选择待检测文件")
        elif os.path.isfile(self.fordetect): # 单张图片
            self.imgDetect(self.fordetect)
        elif os.path.isdir(self.fordetect): # 文件夹
            self.dirDetect(self.fordetect)
        elif self.fordetect == '1': # 调用摄像头数据
            self.my_timer.timeout.connect(self.cameraDetect)
            # 改变按钮状态
            if self.btn_status:
                self.btn_status = False
            else:
                self.btn_status = True

            if self.btn_status:
                self.runButton.setText('暂停')
            else:
                self.runButton.setText('运行')
                self.my_timer.stop()
                self.cap.release()
                self.run_info.setText("运行结束")



    # 检测图片
    def imgDetect(self, file):
        img_source = file
        self.run_info.setText("正在检测图片...")
        result_img, self.result_info = detect_img(source = img_source, project=self.save_dir)

        temp_imgSrc = QImage(result_img, result_img.shape[1], result_img.shape[0], result_img.shape[1] * 3, QImage.Format_BGR888)
        # 将图片转换为QPixmap方便显示
        pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(self.result_img.width(), self.result_img.height())

        self.result_img.setPixmap(pixmap_imgSrc)
        self.show_result_info()
        self.run_info.setText("检测完成")

    # 检测文件夹
    def dirDetect(self, dir):
        self.run_info.setText("正在检测图片...")
        files = os.listdir(dir)
        for file in files:
            filename = dir + '/' +file
            # 展示原图
            img = QtGui.QPixmap(filename).scaled(self.origin_img.width(), self.origin_img.height())
            self.origin_img.setPixmap(img)
            # 检测
            result_img, self.result_info = detect_img(source = filename, project=self.save_dir)
            temp_imgSrc = QImage(result_img, result_img.shape[1], result_img.shape[0], result_img.shape[1] * 3, QImage.Format_BGR888)
            # 将图片转换为QPixmap方便显示
            pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(self.result_img.width(), self.result_img.height())
            self.result_img.setPixmap(pixmap_imgSrc)
            self.show_result_info()
            QApplication.processEvents()  # 刷新界面
            time.sleep(0.1)
        self.run_info.setText("检测完成")

    # 检测摄像头 （先保存再检测图片）
    def cameraDetect(self):
        self.run_info.setText("正在检测图片...")
        imgpath = str(Path(self.save_dir))+'/%s.jpg' % str(time.time())
        self.show = cv2.cvtColor(self.show, cv2.COLOR_RGB2BGR)
        cv2.imwrite(imgpath, self.show)
        # 检测
        result_img, self.result_info = detect_img(source=imgpath, project=self.save_dir)
        temp_imgSrc = QImage(result_img, result_img.shape[1], result_img.shape[0], result_img.shape[1] * 3,
                             QImage.Format_BGR888)
        # 将图片转换为QPixmap方便显示
        pixmap_imgSrc = QPixmap.fromImage(temp_imgSrc).scaled(self.result_img.width(), self.result_img.height())
        self.result_img.setPixmap(pixmap_imgSrc)
        self.show_result_info()
        QApplication.processEvents()  # 刷新界面
        time.sleep(0.1)

    # 选择保存路径
    def saveDir(self):
        self.save_dir_name = QFileDialog.getExistingDirectory(self, '请选择保存路径', './')
        if (self.save_dir_name):
            self.run_info.setText("选择保存路径" + self.save_dir_name);
        else:
            self.run_info.setText("打开路径失败");

    # 在文本框显示检测结果
    def show_result_info(self):
        self.textBrowser_0.setText(self.result_info[0])
        self.textBrowser_1.setText(str(self.result_info[1]))
        self.textBrowser_2.setText(str(self.result_info[2]))
        self.textBrowser_3.setText(str(self.result_info[3]))
        self.textBrowser_4.setText(str(self.result_info[4]))
        self.textBrowser_5.setText(self.result_info[5])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = UiMain()
    ui.show()
    sys.exit(app.exec_())
