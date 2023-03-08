from pantilt import Pantilt
import time

pt = Pantilt()

for x in range(200, 800, 1):
    for y in range(310, 650, 1):
        pt.position(x, y)
        time.sleep(0.01)