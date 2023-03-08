
#POUR CALIBRER LA CAMERA : 
#Impression de la grille de calibration sur feuille (échiquier 6*9)
#Prendre des photos de la grille de calibration avec la caméra (au moins 5 dans des positions différentes)
#les mettre dans le dossier calibration_images
#faire tourner le programme

import cv2
import numpy as np
import glob

# Créer une liste des images de calibration
images = glob.glob('Code projet initial/arducam_64mp/[1-5].jpg')
if len(images)==0:
    print("Aucune image trouvée")
else:   
    print("les images sont : ", images)

    # Définir la taille de la grille de calibration (nombre de points internes)
    pattern_size = (9, 6)

    # Créer un tableau de stockage pour les points de la grille de calibration
    obj_points = []
    img_points = []

    # Générer les coordonnées 3D de la grille de calibration
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    # Parcourir toutes les images de calibration
    for fname in images:
        print("Traitement de l'image : ", fname)
        # Charger l'image et la convertir en niveaux de gris
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Détecter les coins de la grille de calibration dans l'image
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # Si des coins sont trouvés, ajouter les points de la grille de calibration et les points d'image correspondants
        if ret == True:
            obj_points.append(objp)
            img_points.append(corners)

            # Afficher les coins trouvés sur l'image
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    # Fermer la fenêtre d'affichage
    cv2.destroyAllWindows()

    # Obtenir les paramètres de la caméra en effectuant la calibration de caméra
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # Afficher la matrice de la caméra et les coefficients de distorsion
    print("Matrice de la caméra : \n", mtx)
    print("Coefficients de distorsion : \n", dist)