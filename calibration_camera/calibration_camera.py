
#POUR CALIBRER LA CAMERA : 
#Impression de la grille de calibration sur feuille (échiquier 6*9)
#Prendre des photos de la grille de calibration avec la caméra (au moins 5 dans des positions différentes)
#les mettre dans le dossier calibration_images
#faire tourner le programme

import cv2
import numpy as np
import glob
import tkinter as tk
from tkinter import filedialog
import os

# Définir la taille de la grille de calibration (nombre de points internes)
# Si les valeurs fx et fy sont très différentes, peut être que la grille de calibration n'est pas (X, Y) mais (Y, X)
pattern_size = (5, 8)


########CHOIX DES IMAGES DE CALIBRATION############################################################################################################
initial_dir = os.path.dirname(os.path.abspath(__file__)) 
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilenames(initialdir = initial_dir, title = "Sélectionnez les images de calibration", filetypes = (("jpeg files","*.jpg"),("all files","*.*")))

if len(image_path) ==0  :
    print("Aucune image sélectionnée")
else:   
    print("les images sont : ", image_path, "\n")

#######DETERMINATION DES CARACTERISTIQUES DE LA CAMERA#############################################################################

    # Créer un tableau de stockage pour les points de la grille de calibration
    obj_points = []
    img_points = []

    # Générer les coordonnées 3D de la grille de calibration
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    # Parcourir toutes les images de calibration
    for fname in image_path:
        print("Traitement de l'image : ", fname)
        # Charger l'image et la convertir en niveaux de gris
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Détecter les coins de la grille de calibration dans l'image
        ret, corners = cv2.findChessboardCornersSB(img, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE)

        # Si des coins sont trouvés, ajouter les points de la grille de calibration et les points d'image correspondants
        if ret == True:
            obj_points.append(objp)
            img_points.append(corners)

            # Afficher les coins trouvés sur l'image
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)

    # Fermer la fenêtre d'affichage
    cv2.destroyAllWindows()

    # Obtenir les paramètres de la caméra en effectuant la calibration de caméra
    ret, camera_matrix, distorsion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)


#######AFFICHAGE DES RESULTATS#############################################################################

    fx = round(camera_matrix[0][0], 3)
    fy = round(camera_matrix[1][1], 3)
    cx = round(camera_matrix[0][2], 3)
    cy = round(camera_matrix[1][2], 3)
    k1 = round(distorsion_coefficients[0][0], 3)
    k2 = round(distorsion_coefficients[0][1], 3)
    p1 = round(distorsion_coefficients[0][2], 3)
    p2 = round(distorsion_coefficients[0][3], 3)
    k3 = round(distorsion_coefficients[0][4], 3)

    print("\n\n********** Caractéristiques de la caméra**********\n")
    print("\nMatrice caractéristique : |fx = ", fx, "|0\t\t", " |cx = ",cx,
                                    "|\n\t\t\t  |0\t\t  |fy = ", fy, "|cy = ", cy,
                                    "|\n\t\t\t  |0\t\t  |0\t\t", " |1\t\t  |\n")


    print("Coefficients de distorsion : k1 =", k1, "\tk2 =", k2, "\tp1 =", p1, "\tp2 =", p2, "\tk3 =", k3)

    print("\n\n********** Caractéristiques spécifiques à chaque images**********\n")

    for i in range(len(image_path)):
        print("\nImage n°", i+1)
        Tx = round(translation_vectors[i][0][0], 3)
        Ty = round(translation_vectors[i][1][0], 3)
        Tz = round(translation_vectors[i][2][0], 3)
        theta = round(rotation_vectors[i][0][0], 3)
        phi = round(rotation_vectors[i][1][0], 3)
        psi = round(rotation_vectors[i][2][0], 3)

        print("Vecteurs de rotation : \t  \u03B8  =", theta, "\t\u03C6  =", phi, "\t\u03C8  =", psi)
        print("Vecteurs de translation : Tx =", Tx, "\tTy =", Ty, "\tTz =", Tz)
  

    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints2, _ = cv2.projectPoints(obj_points[i], rotation_vectors[i], translation_vectors[i], camera_matrix, distorsion_coefficients)
        error = cv2.norm(img_points[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print ("\n\n\nErreur totale de calibration, plus la valeur est proche de 0, mieux c'est : ", round(mean_error/len(obj_points), 3))
