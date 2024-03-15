import RPi.GPIO as GPIO
import time

# 设置GPIO引脚模式
GPIO.setmode(GPIO.BOARD)

# 定义GPIO引脚
IN1 = 38
IN2 = 36
ENA = 40

# 设置GPIO引脚为输出模式
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

# 控制伸缩杆的伸缩
def extend():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)

def retract():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)

# 启用电机驱动5  
GPIO.output(ENA, GPIO.HIGH)

# 伸缩杆伸缩一段时间后停止
extend()
time.sleep(7)
retract()
time.sleep(7)

# 停用电机驱动
GPIO.output(ENA, GPIO.LOW)

# 清理GPIO引脚设置
GPIO.cleanup()
