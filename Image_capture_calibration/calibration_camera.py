
#POUR CALIBRER LA CAMERA : 
#Impression de la grille de calibration sur feuille (damier noir et blancs
#Prendre des photos de la grille de calibration avec la caméra (au moins 10 dans des positions différentes)
#les mettre dans le dossier calibration_images
#faire tourner le programme

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import datetime

# Définir la taille de la grille de calibration (nombre de points internes)
# Si les valeurs fx et fy sont très différentes, peut être que la grille de calibration n'est pas (X, Y) mais (Y, X)
pattern_size = (8, 8)

# Définir la taille de l'ouverture de la caméra (en mm)
#arducam64mp : 7.4 x 5.56 mm
#arducam16mp : 5.680 x 4.265 mm
#raspberry pi camera v3 : 6.4512 x 4.2768 mm
aperture_width = 5.680 #en mm
aperture_height = 4.265 #en mm

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
    size = gray.shape[::-1]
    print("Taille de l'image : ", size)
    ret, camera_matrix, distorsion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(obj_points, img_points, size, None, None)

    fov_x, fov_y, focal_length, principal_point, aspect_ratio = cv2.calibrationMatrixValues(camera_matrix, size, aperture_width, aperture_height)

    
#######AFFICHAGE DES RESULTATS#############################################################################

    fx = round(camera_matrix[0][0], 5)
    fy = round(camera_matrix[1][1], 5)
    cx = round(camera_matrix[0][2], 5)
    cy = round(camera_matrix[1][2], 5)
    k1 = round(distorsion_coefficients[0][0], 5)
    k2 = round(distorsion_coefficients[0][1], 5)
    p1 = round(distorsion_coefficients[0][2], 5)
    p2 = round(distorsion_coefficients[0][3], 5)
    k3 = round(distorsion_coefficients[0][4], 5)
    fov_x = round(fov_x, 5)
    fov_y = round(fov_y, 5)
    focal_length = round(focal_length, 5)
    aspect_ratio = round(aspect_ratio, 5)

    print("\n\n********** Caractéristiques de la caméra**********\n")
    print("\nMatrice intrinseque :   |fx = ", fx, "|0\t\t", "     |cx = ",cx,
                                    "|\n\t\t\t|0\t\t  |fy = ", fy, "  |cy = ", cy,
                                    "|\n\t\t\t|0\t\t  |0\t\t", "     |1\t\t|\n")


    print("Coefficients de distorsion : k1 =", k1, "\tk2 =", k2, "\tp1 =", p1, "\tp2 =", p2, "\tk3 =", k3)
    print("fov_x = ", fov_x, "\tfov_y = ", fov_y, "\tfocal_length = ", focal_length,"principal point = ", principal_point,  "\taspect_ratio = ", aspect_ratio)

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


######SAUVEGARDE DES RESULTATS#############################################################################

file_save_path = filedialog.asksaveasfilename(initialdir = initial_dir, title = "Sauvegarder les résultats", filetypes = (("text files","*.txt"),("all files","*.*")))

text = ""
actual_time = datetime.datetime.now()
text = text + "Date de la calibration : " + str(actual_time.day) + "/" + str(actual_time.month) + "/" + str(actual_time.year) + " : " + str(actual_time.hour)+ "h" + str(actual_time.minute) +   "min\n\n"

text = text + "********************Images utilisees pour la calibration : \n"
for i in range(len(image_path)):
    text = text + "\t" + image_path[i] + "\n"

text = text + "\n********************Donnees d'entree : \n"
text = text + "\tTaille de l'image : " + str(size[0]) + "x" + str(size[1]) + " [pixels]\n"
text = text + "\tTaille du capteur : " + str(aperture_width) + "x" + str(aperture_height) + " [mm]\n"
text = text + "\tLongueur du damier : " + str(pattern_size[0] + 1) + "\n"
text = text + "\tLargeur du damier : " + str(pattern_size[1] + 1) + "\n"



text = text + "\n********************Resultats de l'etude : \n"
text = text + "\nMatrice intrinseque : \n"
text = text + "\tLongeur focale en x : " + str(fx) + " [pixels]\n"
text = text + "\tLongeur focale en y : " + str(fy) + " [pixels]\n"
text = text + "\tPosition du point principal sur le capteur: X = " + str(cx) + "\tY = " + str(str(cy)) + " [pixels]\n\n"
text = text + "\tCoefficients de distorsion : k1 = " + str(k1) + "\tk2 = " + str(k2) + "\tp1 = " + str(p1) + "\tp2 = " + str(p2) + "\tk3 = " + str(k3) + "\n\n"
text = text + "\tAngle de vue en sortie en X : " + str(fov_x) + " [degres]\n"
text = text + "\tAngle de vue en sortie en Y : " + str(fov_y) + " [degres]\n"
text = text + "\tLongueur focale de la lentille : " + str(focal_length) + " [mm]\n"
text = text + "\tRapport d'aspect : " + str(aspect_ratio) + "\n\n"
text = text + "\tPosition du point principal sur le capteur: X = " + str(round(principal_point[0], 3)) + "\tY = " + str(round(principal_point[1], 3)) + " [mm]\n\n"

text = text + "********************Caracteristiques specifiques pour chaque image : --> matrice extrinseque \n"

for i in range(len(image_path)):
    Tx = round(translation_vectors[i][0][0], 3)
    Ty = round(translation_vectors[i][1][0], 3)
    Tz = round(translation_vectors[i][2][0], 3)
    theta = round(rotation_vectors[i][0][0], 3)
    phi = round(rotation_vectors[i][1][0], 3)
    psi = round(rotation_vectors[i][2][0], 3)
    text = text + "\nImage " + str(i+1) + ":"
    text = text + "\n\tVecteurs de rotation [pixels] : \t  theta  = " + str(theta) + "\t\tphi  = " + str(phi) + "\t\tpsi  = " + str(psi) 
    text = text + "\n\tVecteurs de translation [pixels] : Tx = " + str(Tx) + "\t\t\tTy = " + str(Ty) + "\t\t\tTz = " + str(Tz) + "\n"

try :
    with open(file_save_path + ".txt", "w") as f:
        f.write(text)
except:
    print("Erreur dans l'enregistrement du ficher")
