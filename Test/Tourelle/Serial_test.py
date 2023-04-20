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
    nb = randint(1,3)

    if nb == 1:
        x = 0
        y = 0
        f = randint(2,10)
        for k in range (0,f):
            for y in range (13,90,1):
                xmap = pt.map(90,0,180,200,800)
                ymap = pt.map(y,0,90,310,650)
                pt.position(xmap,ymap)
                time.sleep(0.005)

    elif nb ==2:
        x = 0
        y = 0
        f = randint(2,10)
    
        for k in range (0,f):
            for x in range (0,180,1):
                xmap = pt.map(x,0,180,200,800)
                ymap = pt.map(y,0,90,310,650)
                pt.position(xmap,ymap)
                time.sleep(0.002)


    elif nb == 3:

        x = 0
        y = 0
        f = randint(2,10)
    
        for k in range (0,f):
            for x in range (0,90,1):
                xmap = pt.map(x*2,0,180,200,800)
                ymap = pt.map(x,0,90,310,650)
                pt.position(xmap,ymap)
                time.sleep(0.002)
    

   
    

    
    

    

"""
while True:
    x = int(input("x: "))
    y = int(input("y: "))

    xmap = pt.map(x,0,180,200,800)
    ymap = pt.map(y,0,90,310,650)

    pt.position(xmap,ymap)
"""
    
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