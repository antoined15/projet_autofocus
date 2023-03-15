import os
import sys
import time
from picamera2 import Picamera2
from libcamera import controls
from PIL import Image

def get_next_filename():
    i = 1
    while os.path.exists(f'image_{i}.jpg'):
        i+=1
    return f'image_{i}.jpg'

def take_photo(camera):
    filename = get_next_filename()
    frame = camera.capture_image()
    frame.convert('RGB').save(filename)
    print(f'Photo prise et sauvegardée sous le nom {filename}')

def main():
    with Picamera2() as camera:
        camera.configure(camera.create_preview_configuration(main={"size":(1920,1080)}))
        camera.start(show_preview=True)
        camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        time.sleep(2)

        print("Appuyez sur 'entree' pour prendre une photo, ou 'q' pour quitter.")
        while True:
            key = input()
            if key.lower() == 'q':
                break
            take_photo(camera)
        
        camera.stop()

if __name__ == '__main__':
    main()
