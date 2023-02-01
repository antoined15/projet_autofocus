import numpy as np
import cv2
cap = cv2.VideoCapture(0)


def variance_of_image_blur_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	return round(cv2.Laplacian(image, cv2.CV_64F).var(), 2)

def variance_of_image_blur_sobel(image):
	# compute the Sobel of the image and then return the focus
	return round(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3 ).var(), 1)

def variance_of_image_blur_Canny(image):
	# compute the Canny of the image and then return the focus
	return round(cv2.Canny(image, 100, 200).var(), 1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    # Operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fm_sobel = variance_of_image_blur_sobel(gray)
    fm_laplacian = variance_of_image_blur_laplacian(gray)
    fm_canny = variance_of_image_blur_Canny(gray)

    # Display the resulting frame
    cv2.putText(frame, "{}{:.2f}".format(" Sobel focus value : ", fm_sobel), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "{}{:.2f}".format(" Laplacian focus value : ", fm_laplacian), (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, "{}{:.2f}".format(" Canny focus value : ", fm_laplacian), (0, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    
    
    
    # Appliquer un seuil pour traiter l'image *****************************************************
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) #amélioré
    #cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY) #simplifié

    # Trouver les contours de la mire*********************************************************
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Boucle sur les contours pour trouver la mire
    for cnt in contours:
        #cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 1) affiche tous les countours
        # Si le contour a suffisamment de points
        if len(cnt) >= 5:
            # Calculer les moments
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                # Calculer les coordonnées de la mire
                cont_x = int(M["m10"] / M["m00"])
                cont_y = int(M["m01"] / M["m00"])
                 # Afficher les contours sur l'image originale
                #cv2.circle(frame, (cont_x, cont_y), 20, (0, 0, 255), 2)


    
    # Trouver les rectangles de la mire*********************************************************
    for cnt in contours:

        # Vérifier si le contour est suffisamment grand pour être considéré comme un rectangle
        if cv2.contourArea(cnt) < 1:
            continue
        # Obtenir le rectangle minimum qui englobe les points de contour
        rect = cv2.minAreaRect(cnt)

        (x, y), (w, h), angle = rect
        ratio = h/w
        if 0.05 < ratio < 0.6: #si on a a pe pres le ratio des rectangles de la mire : 1/5 = 0.2
            # Dessiner le rectangle sur l'image
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame,[box],0,(255,0,),2)

    # Afficher l'image*********************************************************************
    cv2.imshow('Focus value', frame)
    if cv2.waitKey(1) == ord('q'): #waitkey(0) --> only one frame, waitkey(1) --> video 
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()