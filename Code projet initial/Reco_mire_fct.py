#Ce fichier contient des fonctions utilisées par Reco_mire.py

import cv2
import numpy as np
import math


def variance_of_image_blur_laplacian(gray_image): 
        #Calcule la valeur de flou de l'image avec le Laplacien
	return round(cv2.Laplacian(gray_image, cv2.CV_64F).var(), 2)

def variance_of_image_blur_sobel(gray_image): 
    #Calcule la valeur de flou de l'image avec le Sobel --> méthode de la dérivée seconde
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5) #calcul du gradient de Sobel selon l'axe x
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5) #calcul du gradient de Sobel selon l'axe y
    return cv2.norm(sobelx, sobely, cv2.NORM_L2) #calcul de la norme des deux gradients

def variance_of_image_blur_Canny(gray_image):
    #Calcule la valeur de flou de l'image avec l'algorithme de Canny
	return round(cv2.Canny(gray_image, 100, 200).var(), 1)

def variance_of_image_blur_entropy(gray_image):
    #Calcule la valeur de flou de l'image par méthode statistique avec l'entropie : dispersion des pixels de l'image
    hist = np.histogram(gray_image, bins=256, range=(0, 255))[0]
    hist = hist / hist.sum()
    return -np.sum(hist * np.log2(hist + 1e-7))

def variance_of_image_blur_corner_counter(gray_image): 
    #Calcule la valeur de flou de l'image avec le nombre de coins détectés
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(gray_image, None)
    return len(kp)

def variance_of_image_blur_picture_variability(gray_image): 
    #Calcule la valeur de flou de l'image avec la variabilité de l'image
    mean , stddev = cv2.meanStdDev(gray_image)
    return mean[0][0]


def perspective_mire(image, box): 
    #fonction pour transformer l'image de mire en perspective --> Sert à rien en tant que tel   
    pts1 = np.float32([box[0], box[1], box[3], box[2]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])        

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective = cv2.warpPerspective(image, matrix, (300, 300))
    cv2.imshow('perspective', perspective)


def  matrice_normalisation(matrice_inputs, nbr_col_matrice, nbr_row_matrice ): 
    #Fonction qui permet de normaliser la matrice d'entrée pour placer les symboles dans une matrice de taille nbr_row_matrice*nbr_col_matrice 

    normalised_points_x = [point[0] for point in matrice_inputs]
    normalised_points_y = [point[1] for point in matrice_inputs]

    min_x = np.min(normalised_points_x)
    min_y = np.min(normalised_points_y)
    max_x = np.max(normalised_points_x)
    max_y = np.max(normalised_points_y)

    normalised_points_x = ((normalised_points_x - min_x)/(max_x - min_x)) * (nbr_row_matrice-1) 
    normalised_points_y = ((normalised_points_y - min_y)/(max_y - min_y)) * (nbr_col_matrice-1) 

    normalised_points = np.transpose(np.array([normalised_points_x, normalised_points_y]))

    return normalised_points

def matrice_rotation(angle_deg, mean_x, mean_y, matrice): 
    #fonction qui permet de tourner la matrice de symbole pour qu'elle soit bien orientée en fonction de l'angle de la mire détecté
    angle_rad = math.radians(-angle_deg) #on convertit l'angle en radians
    rotation_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]]) #on crée la matrice de rotation
    centre_matrix = np.array([mean_x, mean_y])
    rotated_matrice = []
    for point in matrice:
        points = point - centre_matrix
        rotated_point = np.dot(rotation_matrix, points)
        rotated_matrice.append(rotated_point)
    return rotated_matrice



def pourcent_symbole_detected(matrice): 
    #fonction qui permet de calculer le pourcentage de symbole détecté dans la matrice de symbole
    nbr_row_matrice = np.size(matrice, 0)
    nbr_col_matrice = np.size(matrice, 1)

    symbole_found = np.count_nonzero(np.array(matrice))
    pourcent_symb_detect = round(100 * symbole_found/(nbr_row_matrice*nbr_col_matrice), 3)
    return pourcent_symb_detect

def matrice_rgb_show(matrice): #transforme la matrice numérique en matrice couleur et l'affiche
    
    nbr_row_matrice = np.size(matrice, 0)
    nbr_col_matrice = np.size(matrice, 1)

    matrice_rgb = np.zeros((nbr_row_matrice, nbr_col_matrice, 3), dtype=np.uint8)
    colors = [(0, 0, 0), (0, 0, 255), (255,0 , 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]
    for i in range(nbr_row_matrice):
        for j in range(nbr_col_matrice):
            matrice_rgb[j, i] = colors[matrice[i, j]]
            try : 
                matrice_rgb[j, i] = colors[matrice[i, j]]
            except:
                matrice_rgb[i, j] = (0, 0, 0)
    matrice_rgb = cv2.resize(matrice_rgb, (450, 450), cv2.INTER_NEAREST)
    pourc_symb = pourcent_symbole_detected(matrice) #affiche le pourcentage de symboles détecté
    cv2.putText(matrice_rgb, "{}{}{}".format(" Symboles detectes : ", round(pourc_symb, 2), "%"), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('matrice des symboles detectes', matrice_rgb) #on affiche la matrice de symbole


def symbole_identique_2_matrices_V3(matrice1, matrice2): #calcul de l'erreur quadratique moyenne entre deux matrices
    h, w = np.array(matrice1).shape
    diff = matrice1 - matrice2
    err = np.sum(diff**2)
    mse = err/float((h*w))
    return round(mse, 5)



def symbole_identique_2_matrices_V4(matrice1, matrice2): #calcul de l'erreur entre 2 matrices et ne pénalise pas si un symbole n'est pas détecté
    juste = 0

    nbr_row_matrice = np.size(matrice1, 0)
    nbr_col_matrice = np.size(matrice1, 1)
    
    for i in range(nbr_row_matrice):
        for j in range(nbr_col_matrice):
            if matrice2[i][j] !=0: #si le symbole n'est pas détecté, on passe
                if matrice1[i][j] == matrice2[i][j]:
                    juste = juste + 1

    pourcent_erreur = 100 * (1-(juste / (nbr_row_matrice*nbr_col_matrice)))       
    return np.round(pourcent_erreur, 5)


def comparison_mire(real_mire, matrice_symb): #fonction globale pour comparer la mire originelle avec la mire potentielle trouvée
    #on va tourner la matrice de symbole pour qu'elle soit bien orientée en fonction de l'angle de la mire détecté

    matrice_symb_rot_90 = np.rot90(matrice_symb, k=1)
    matrice_symb_rot_180 = np.rot90(matrice_symb, k=2)
    matrice_symb_rot_270 = np.rot90(matrice_symb, k=3)

    pourcent_symb_valide_0 = symbole_identique_2_matrices_V4(real_mire, matrice_symb)
    pourcent_symb_valide_90 = symbole_identique_2_matrices_V4(real_mire, matrice_symb_rot_90)
    pourcent_symb_valide_180 = symbole_identique_2_matrices_V4(real_mire, matrice_symb_rot_180)
    pourcent_symb_valide_270 = symbole_identique_2_matrices_V4(real_mire, matrice_symb_rot_270)

    #print("Mean Square Error : lower the value, better the shape")
    #print("r=0° : ",pourcent_symb_valide_0,)
    #print("r=90° : ",pourcent_symb_valide_90)
    #print("r=180° : ",pourcent_symb_valide_180)
    #print("=270° : ",pourcent_symb_valide_270)

    max_symb_valid = min(pourcent_symb_valide_0, pourcent_symb_valide_90, pourcent_symb_valide_180, pourcent_symb_valide_270)
    if max_symb_valid == pourcent_symb_valide_0:
        rot_probable = 0
    elif max_symb_valid == pourcent_symb_valide_90:
        rot_probable = 90
    elif max_symb_valid == pourcent_symb_valide_180:
        rot_probable = 180
    else:
        rot_probable = 270
    #print("Rotation de la mire probable :", rot_probable, "°")
    #print("----------------")
    return rot_probable

def detect_contours(frame, contour_show = False):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convertir en noir et blanc
    canny_image = cv2.Canny(gray, 50,150) #Appliquer un filtre de Canny pour détecter les contours
    contours, _ = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #trouver les contours de l'image avec le filtre de Canny
    canny_image_cont = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB) #convertir en couleur pour pouvoir afficher les contours en couleur --> frame avec contours
    if (contour_show):
        cv2.imshow('Contours', canny_image_cont) #on affiche l'image avec les contours   

    return contours


