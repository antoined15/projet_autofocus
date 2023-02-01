
import numpy as np
import cv2
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(0)

while True: 
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #amélioré

    canny_image = cv2.Canny(frame, 50, 150) #trouver les contours avec canny #méthode améliorée pour détecter les contours 


     #Affiche les rectangles détectés       
    contours, _ = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #trouver les contours
    canny_image_rgb = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)
    n = 0
    long = 0
    for cnt in contours:     #détection des traits
        cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 1) #affiche tous les countours

        # Vérifier si le contour est suffisamment grand pour être considéré comme une forme
        if cv2.contourArea(cnt) > 5:
            # Obtenir le rectangle minimum qui englobe les points de contour
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            ratio = h/w


            if 0.2 < ratio <0.35 or 2 < ratio <10:  #les rectangles sont environ 5fois plus grand en longueur que el largeur : ratio = env 3.5 ou env 0.28

                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(canny_image_rgb,[box],0,(255,0,255),2)
                #longeur de la ligne
 
                #long = max(w, h) + long
                #n = n+1
            if 0.9 < ratio <1.1:  #détection des cercles : on considère que c'est des cercles parfaits

                box = cv2.boxPoints(rect)
                box = np.int0(box)
                #cv2.drawContours(canny_image_rgb,[box],0,(0,0,255),2)
                cv2.circle(canny_image_rgb, (int(x), int(y)), int(min(w/2,h/2)), (255, 255, 0), 2)




   # if n>0:
        #moy_diam = long/n
       # print(moy_diam)
        #n=0
        #long=0
        #max_diam = round(moy_diam*1)



   # circles = cv2.HoughCircles(canny_image, cv2.HOUGH_GRADIENT, dp = 0.1, minDist = max_diam/2 , param1=1, param2=15, minRadius=0, maxRadius=max_diam)

    # Tracer les cercles détectés sur l'image
    #if circles is not None:
       # for circle in circles[0]:
            #if circle is not None :
                # Coordonnées x et y du centre du cercle
                #x, y = int(circle[0]), int(circle[1])
                # Rayon du cercle
               #r = int(circle[2])
                # Tracer le cercle sur l'image
                #cv2.circle(canny_image_rgb, (x, y), r, (0, 255, 0), 2)


  

    #Affiche l'image'
    cv2.imshow('principal frame', frame)
    cv2.imshow('edge detection', canny_image_rgb)
    if cv2.waitKey(1) == ord('q'): #waitkey(0) --> only one frame, waitkey(1) --> video 
        break


cap.release()
cv2.destroyAllWindows()