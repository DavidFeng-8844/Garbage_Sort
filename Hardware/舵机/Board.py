import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)
GPIO.setup(16, GPIO.OUT)
GPIO.setup(18, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)
GPIO.setup(24, GPIO.OUT)


db1=GPIO.PWM(16, 50)
db2=GPIO.PWM(18, 50)
db3=GPIO.PWM(22, 50)
db4=GPIO.PWM(24, 50)             
db1.start(0)
db2.start(0)
db3.start(0)
db4.start(0)
while True:
    print('Board open')
    db1.ChangeDutyCycle(5.4) 
    db2.ChangeDutyCycle(5.5)
    db3.ChangeDutyCycle(4.4) 
    db4.ChangeDutyCycle(6) 
    sleep(2)

    print('Board close')
    db1.ChangeDutyCycle(7.2) 
    db2.ChangeDutyCycle(8.2)
    db3.ChangeDutyCycle(6.5) 
    db4.ChangeDutyCycle(8.1) 
    sleep(2)

print('hhh')
