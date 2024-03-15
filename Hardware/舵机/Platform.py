import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BOARD)
GPIO.setup(12, GPIO.OUT)
GPIO.setup(10, GPIO.OUT)

up=GPIO.PWM(12, 50)
ba=GPIO.PWM(10, 50)
up.start(0)
ba.start(0)
while True:
    print('Positon 0')
    up.ChangeDutyCycle(5.5) # Horizontal
    ba.ChangeDutyCycle(5.2) # Parallel to y 
    sleep(2) 
'''
    print('Positon 1')
    up.ChangeDutyCycle(4.5) # Horizontal
    ba.ChangeDutyCycle(7.5) # Parallel to y 
    sleep(2)

    print('Positon 0')
    up.ChangeDutyCycle(6.8) # Horizontal
    ba.ChangeDutyCycle(5.2) # Parallel to y 
    sleep(2)'''
'''
    print('Positon 2')
    up.ChangeDutyCycle(2) # Horizontal
    ba.ChangeDutyCycle(2) # Parallel to y 
    sleep(2)

    print('Positon 0')
    up.ChangeDutyCycle(5) # Horizontal
    ba.ChangeDutyCycle(4.5) # Parallel to y 
    sleep(2)

    print('Positon 3')
    up.ChangeDutyCycle(2) # Horizontal
    ba.ChangeDutyCycle(7) # Parallel to y 
    sleep(2)

    print('Positon 0')
    up.ChangeDutyCycle(5) # Horizontal
    ba.ChangeDutyCycle(4.5) # Parallel to y 
    sleep(2)

    print('Positon 4')
    up.ChangeDutyCycle(7.3) # Horizontal
    ba.ChangeDutyCycle(2) # Parallel to y 
    sleep(2)

    print('Positon 0')
    up.ChangeDutyCycle(5) # Horizontal
    ba.ChangeDutyCycle(4.5) # Parallel to y 
    sleep(2)
'''
print('hhh')
