import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)

test_pin = 22

GPIO.setup(test_pin, GPIO.OUT)
test_motor=GPIO.PWM(test_pin, 50)

test_motor.start(0)

while True:
    print('state 1')

    test_motor.ChangeDutyCycle(7) 

    sleep(2)

    print('state 2')

    test_motor.ChangeDutyCycle(9) 

    sleep(2)

print('End')
