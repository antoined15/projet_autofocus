
#ecriture sur le port série


# Module de lecture/ecriture du port série
from serial import *
ports = serial.tools.list_ports(serial.tools.list_ports.comports())
for p in ports:
    print(p)


