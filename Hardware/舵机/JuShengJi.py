import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)

GPIO.setup(22, GPIO.OUT)



db3=GPIO.PWM(22, 50)

db3.start(0)

while True:
    print('open')

    db3.ChangeDutyCycle(8.5) 

    sleep(2)

    print('close')

    db3.ChangeDutyCycle(4.6) 

    sleep(2)

print('hhh')
