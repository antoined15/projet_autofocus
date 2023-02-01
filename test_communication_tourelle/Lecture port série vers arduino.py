# Module de lecture/ecriture du port série
from serial import *
import serial.tools.list_ports as list_ports
ports = list(list_ports.comports())
print(ports)
for p in ports:
    print(ports)

