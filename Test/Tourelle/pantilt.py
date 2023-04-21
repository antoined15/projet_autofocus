import serial
import numpy as np
import time

# '/dev/ttyUSB0', 57600

class Pantilt:
    """Class to control the pantilt"""

    __x = 0
    __y = 0

    """Constructor of the class Pantilt"""
    def __init__(self, com_port = '/dev/ttyUSB0', com_speed = 57600):
        self.com_port = com_port
        self.com_speed = com_speed

    """Destructor of the class Pantilt"""
    def __del__(self):
        self.disconnect()

    """Get the last position of the pantilt"""
    def get_last_x(self):
        return self.__x
    
    """Get the last position of the pantilt"""
    def get_last_y(self):
        return self.__y
    
    """Change the port and speed of the pantilt"""
    def change_port(self, com_port, com_speed):
        self.com_port = com_port
        self.com_speed = com_speed

    """Connect to the pantilt"""
    def connect(self):
        try:
            self.ser = serial.Serial(self.com_port, self.com_speed, timeout=1)
            self.init()
        except:
            print(f'Error opening serial port {self.com_port}')
            return -1

        return 1

    def init(self):
        x = [0,180,90]
        y = [0,90,13]

        for i in x:
            for j in y:
                xmap = self.map(i,0,180,200,800)
                ymap = self.map(j,0,90,310,650)
                self.position(xmap,ymap)
                time.sleep(0.3)
        

    """Disconnect from the pantilt"""
    def disconnect(self):
        try:
            self.ser.close()
        except:
            print(f'Error closing serial port {self.com_port}')
            return -1
        
        return 1

    def map(self, x, in_min, in_max, out_min, out_max):
        return int(np.ceil((x-in_min)*(out_max-out_min)/(in_max-in_min))+out_min)

    """Send a position to the pantilt"""
    def position(self, x, y):
        
        # Checking limits
        if x > 800:
            x = 800
        if y > 650:
            y = 650
        if x < 200:
            x = 200
        if y < 310:
            y = 310
        
        self.__x = x
        self.__y = y
    
        if x < 10:
            pos_x = '000' + str(x)
        elif x < 100:
            pos_x = '00' + str(x)
        elif x < 1000:
            pos_x = '0' + str(x)
        else:
            pos_x = str(x)
        
        if y < 10:
            pos_y = '000' + str(y)
        elif y < 100:
            pos_y = '00' + str(y)
        elif y < 1000:
            pos_y = '0' + str(y)
        else:
            pos_y = str(y)

        trame = 'S' + pos_x + pos_y + '.'
        self.ser.write(trame.encode())
