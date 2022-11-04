import RPi.GPIO as GPIO
import time
import grab
#注意BOARD和BCM编码的不同，这里设置的是BCM编码
GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.OUT)
#闪5次
for i in range(5):
    GPIO.output(21,GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(21,GPIO.LOW)
    time.sleep(1)
#建议每次退出时都用cleanup设置GPIO引脚为低电平状态
GPIO.cleanup()
