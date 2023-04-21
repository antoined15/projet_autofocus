import argparse
from picamera2 import Picamera2
from pantilt import Pantilt
from libcamera import controls
import os
import time
import io
#ce projet c'est du pipo
parser = argparse.ArgumentParser(description="Programme pour effectuer des prise d'images")

parser.add_argument("-d","--device", type=str, default="64mp" ,choices=["16mp","64mp"], required=True, help="Camera utilise [64mp] or [16mp]")
parser.add_argument("-o","--output-dir", type=str, default="image", help="Dossier de destination pour les images")

args = parser.parse_args()

Position_tourelle = [[90, 13], [90,45], [90,0], [90,30], [55,13], [125,13], [70,30], [110,30]] # [X° Y°]
distance = 5

os.system("mkdir "+args.output_dir)

pt = Pantilt()

if pt.connect() == -1:
	print("Impossible de connecter la tourelle pan-tilt")
	exit()

picam2 = Picamera2()
capture_config = picam2.create_preview_configuration(main={"size":(2312,1736)})
picam2.configure(capture_config)
picam2.start(show_preview=True)
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

for pos in Position_tourelle:
    X = pos[0]
    Y = pos[1]

    xmap = pt.map(X,0,180,200,800)
    ymap = pt.map(Y,0,90,310,650)

    for i in range(5):
        pt.position(xmap,ymap)
        time.sleep(0.1)
    time.sleep(2)
    print("Tourelle déplacée : \tmoteur 1 = ", X, " ; \tmoteur 2 = ", Y)

    time.sleep(10)
    picam2.capture_file(name = "main",file_output=f'{args.output_dir}/{args.device}_image_({X},{Y})_{distance}.jpg')
