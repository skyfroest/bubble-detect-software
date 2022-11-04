#coding=utf-8
import datetime
import os
from ctypes import *

import cv2
import numpy as np

import mvsdk

def grab_img():
	# 枚举相机
	DevList = mvsdk.CameraEnumerateDevice()
	nDev = len(DevList)
	if nDev < 1:
		print("No camera was found!")
		return
		
	for i, DevInfo in enumerate(DevList):
		print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
	i = 0 if nDev == 1 else int(input("Select camera: "))
	DevInfo = DevList[i]
	# print(DevInfo)

	# 打开相机
	hCamera = 0
	try:
		hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
	except mvsdk.CameraException as e:
		print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
		return

	# 获取相机特性描述
	cap = mvsdk.CameraGetCapability(hCamera)
	# PrintCapbility(cap)

	# 判断是黑白相机还是彩色相机
	monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

	# 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
	if monoCamera:
		mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)

	# 相机模式切换成连续采集
	mvsdk.CameraSetTriggerMode(hCamera, 0)

	# 手动曝光，曝光时间30ms
	mvsdk.CameraSetAeState(hCamera, 0)
	mvsdk.CameraSetExposureTime(hCamera, 1 * 1000) # 曝光时间设置为1ms

	# 设置图像大小 #这种定义形式，是在python中调用C语言的类型
	sRoiReslution = mvsdk.tSdkImageResolution() # 实例化变量
	sRoiReslution.iIndex = 0xff # 赋值
	sRoiReslution.iWidth = 512
	sRoiReslution.iWidthFOV = 512
	sRoiReslution.iHeight = 512
	sRoiReslution.iHeightFOV = 512
	mvsdk.CameraSetImageResolution(hCamera, sRoiReslution)

	# 让SDK内部取图线程开始工作
	mvsdk.CameraPlay(hCamera)

	# 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
	FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

	# 分配RGB buffer，用来存放ISP输出的图像
	# 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
	pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

	# 从相机取一帧图片
	try:
		pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 2000)
		mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
		mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
		
		# 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
		# 该示例中我们只是把图片保存到硬盘文件中
		timenow = datetime.datetime.now()
		img_name = os.getcwd()+"/camera_images/%s" % timenow
		status = mvsdk.CameraSaveImage(hCamera, img_name, pFrameBuffer, FrameHead, mvsdk.FILE_JPG, 100) # 修改保存图片的类型 JPG\PNG\BMP
		if status == mvsdk.CAMERA_STATUS_SUCCESS:
			print("Save image successfully. image_size = {}X{}".format(FrameHead.iWidth, FrameHead.iHeight) )
		else:
			print("Save image failed. err={}".format(status) )

		# 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
		# frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
		# frame = np.frombuffer(frame_data, dtype=np.uint8)
		# frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))

		# frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
		# cv2.imshow("Press q to end", frame)
		# return frame
		# print(frame) # 如果成功采集图像，则返回该图像

	except mvsdk.CameraException as e:
		print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )

	# 关闭相机
	mvsdk.CameraUnInit(hCamera)

	# 释放帧缓存
	mvsdk.CameraAlignFree(pFrameBuffer)
	# return frame # 之后考虑一下未成功采集图像时，返回什么
	return img_name

def PrintCapbility(cap):
	for i in range(cap.iTriggerDesc):
		desc = cap.pTriggerDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iImageSizeDesc):
		desc = cap.pImageSizeDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iClrTempDesc):
		desc = cap.pClrTempDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iMediaTypeDesc):
		desc = cap.pMediaTypeDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iFrameSpeedDesc):
		desc = cap.pFrameSpeedDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iPackLenDesc):
		desc = cap.pPackLenDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iPresetLut):
		desc = cap.pPresetLutDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iAeAlmSwDesc):
		desc = cap.pAeAlmSwDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iAeAlmHdDesc):
		desc = cap.pAeAlmHdDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iBayerDecAlmSwDesc):
		desc = cap.pBayerDecAlmSwDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )
	for i in range(cap.iBayerDecAlmHdDesc):
		desc = cap.pBayerDecAlmHdDesc[i]
		print("{}: {}".format(desc.iIndex, desc.GetDescription()) )

grab_img()
