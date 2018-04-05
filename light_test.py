import time
import serial

ser = serial.Serial('/dev/cu.usbserial-DN01AGKS')
ser.baudrate = 9600

while True:

    ser.write(1)
    time.sleep(1)

ser.close()
