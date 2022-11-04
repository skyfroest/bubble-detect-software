import RPi.GPIO as GPIO
import time
import datetime
import cv2
from grab import grab_img
from detect import detect_img

#注意BOARD和BCM编码的不同，这里设置的是BCM编码
GPIO.cleanup()
GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.OUT)
#闪5次
for i in range(5):
    GPIO.output(21,GPIO.HIGH)
    time.sleep(0.5)
    img_name = grab_img()
    print(img_name)
    time.sleep(0.5)
    result_img, result_info = detect_img(source = img_name, project='./runs')
    # timenow = datetime.datetime.now()
    # cv2.imwrite('./images/%s.jpg' % timenow, img)
    GPIO.output(21,GPIO.LOW)
    time.sleep(1)
#建议每次退出时都用cleanup设置GPIO引脚为低电平状态
GPIO.cleanup()
