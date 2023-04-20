from pantilt import Pantilt
import time
from random import randint

pt = Pantilt()

if pt.connect() == -1:
    exit()


"""
while True:
    x = randint(200,800)
    y = randint(310,650)

    pt.position(x, y)
    time.sleep(1)
"""

while True:
    x = int(input("x: "))
    y = int(input("y: "))

    xmap = pt.map(x,0,180,200,800)
    ymap = pt.map(y,0,90,310,650)

    pt.position(xmap,ymap)

    
"""
for x in range(0, 180, 1):
    for y in range(0, 90, 1):
        xmap = pt.map(x,0,180,200,800)
        ymap = pt.map(y,0,90,310,650)

        print(xmap)
        print(ymap)

        pt.position(xmap, ymap)
        time.sleep(0.02)
"""