#Ce fichier contient des fonctions utilisées par Reco_mire.py

import cv2
import numpy as np
import math
import threading
import itertools

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

def matrice_rgb_show(matrice, appariements, frame_orig, erreur_moy_appariement, affich_appariement): #transforme la matrice numérique en matrice couleur et l'affiche
    
    nbr_row_matrice = np.size(matrice, 0)
    nbr_col_matrice = np.size(matrice, 1)
    matrice_rgb = np.zeros((450, 450, 3), dtype=np.uint8)

    colors = [(0, 0, 0), (0, 0, 180), (180,0 , 0), (0, 180, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]
    for i in range(nbr_row_matrice):
        for j in range(nbr_col_matrice):

            #on crée des rectangles de couleur pour les symboles détectés
            if matrice[i, j] != 0:
                cv2.rectangle(matrice_rgb, (i*30, j*30), (i*30+30, j*30+30), colors[matrice[i, j]], -1)

    #on affiche les erreurs d'appariement
    if affich_appariement == True and appariements is not None and len(appariements) != 0:
        for appar in appariements : 
            coord_x = int(appar[0][0]*30 + 7)
            coord_y = int(appar[0][1]*30 + 20)
            erreur = str(appar[3])
            cv2.putText(matrice_rgb, erreur, (coord_x,coord_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if affich_appariement == True and appariements is not None and len(appariements) != 0: 
        cv2.putText(frame_orig, "{}{}".format(" Nombre d'appariements trouves : ", len(appariements)), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  
        cv2.putText(frame_orig, "{}{}".format(" Erreur moyenne par d'appariement [symboles/squence] : ", round(erreur_moy_appariement, 2)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  
    pourc_symb = pourcent_symbole_detected(matrice) #affiche le pourcentage de symboles détecté
    cv2.putText(matrice_rgb, "{}{}{}".format(" Symboles detectes : ", round(pourc_symb, 2), "%"), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if affich_appariement:
        affich_img = stack_img(matrice_rgb, frame_orig, type_stack = "horizontal") #on affiche l'image de la matrice de symbole
    else:
        affich_img = matrice_rgb
    #affichage des lignes entre la matrice symbole et l'image de la mire
    if appariements is not None and affich_appariement == True and len(appariements) != 0:
        for appar in appariements : 
            coord_x = int(appar[0][0]*30 + 15)
            coord_y = int(appar[0][1]*30 + 15)
            coord_x2 = int(appar[1][0] + 450)
            coord_y2 = int(appar[1][1])
            cv2.line(affich_img, (coord_x, coord_y), (coord_x2, coord_y2), (0, 70, 255), 1)

    return affich_img


def stack_img(image1, image2, type_stack): #fonction qui permet de superposer deux images soit sur la longueur, soit sur la largeur

    # Convertir les images en couleur si elles sont en niveaux de gris
    try : image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    except : pass
    try : image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    except : pass

    hauteur1, largeur1, canaux1 = image1.shape  
    hauteur2, largeur2, canaux2 = image2.shape

    # Trouver la différence de taille entre les deux images
    diff_hauteur = abs(hauteur1 - hauteur2)
    diff_largeur = abs(largeur1 - largeur2)

    # Créer un cadre vide autour de l'image plus petite
    if type_stack == "horizontal":
        if hauteur1 < hauteur2:
            image1 = cv2.copyMakeBorder(image1, 0, diff_hauteur, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            image2 = cv2.copyMakeBorder(image2, 0, diff_hauteur, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if type_stack == "vertical":
        if largeur1 < largeur2:
            image1 = cv2.copyMakeBorder(image1, 0, 0, 0, diff_largeur, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            image2 = cv2.copyMakeBorder(image2, 0, 0, 0, diff_largeur, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    # Coller les deux images sur la longueur ou la largeur
    if type_stack == "horizontal":
        return  np.hstack((image1, image2))
    elif type_stack == "vertical":
        return np.vstack((image1, image2))
    else:
        return image1

def detect_contours(img):
    #détection des contours de l'image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #conversion de l'image en niveau de gris
    canny_image = cv2.Canny(gray, 50,150) #Appliquer un filtre de Canny sur l'image
    contours, _ = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #Détection des contours 
    
    return contours, canny_image


def suppr_symboles_detectes_non_pertinents (points, points_et_type_et_rayon, treshold):

    mean = np.mean(points, axis=0) #calcul de la moyenne des coordonnées des symboles
    distance = np.linalg.norm(points - mean, axis=1) #calcul de la distance entre chaque point et la moyenne
    good_points_et_type_et_rayon = points_et_type_et_rayon[distance < treshold] #on garde les points qui sont à moins de la distance treshold de la moyenne
    good_points = points[distance < treshold] 

    return good_points, good_points_et_type_et_rayon



###################################################fonctions pour les séquences de symboles#######################################################
def determination_sequences_mire_reel(mire):
    posit_secu = []
    for i in range(1, np.size(mire,0) -1):
        for j in range(1, np.size(mire,1) -1):
                #print("position", i, j)
                A = mire[i][j]
                B = mire[i-1][j]
                C = mire[i][j+1]
                D = mire[i+1][j]
                E = mire[i][j-1]
                F = mire[i-1][j-1]
                G = mire[i-1][j+1]
                H = mire[i+1][j+1]
                I = mire[i+1][j-1]
                sequ = [A, B, C, D, E, F, G, H, I]
                posit_secu_uniq = [i, j], sequ
                posit_secu.append(posit_secu_uniq)

    return posit_secu

def determination_sequences_mire_detect(mire):

    posit_secu = []
    for i in range(1, np.size(mire,0) -1):
        for j in range(1, np.size(mire,1) -1):
                A = int(mire[[i][0]][[j][0]][0])
                B = int(mire[[i-1][0]][[j][0]][0])
                C = int(mire[[i][0]][[j+1][0]][0])
                D = int(mire[[i+1][0]][[j][0]][0])
                E = int(mire[[i][0]][[j-1][0]][0])
                F = int(mire[[i-1][0]][[j-1][0]][0])
                G = int(mire[[i-1][0]][[j+1][0]][0])
                H = int(mire[[i+1][0]][[j+1][0]][0])
                I = int(mire[[i+1][0]][[j-1][0]][0])
                sequ = [A, B, C, D, E, F, G, H, I]
                posit_secu_uniq = [int(mire[i][j][1]), int(mire[i][j][2])], sequ  #mettre les bonnes coordonnées
                if A != 0: #on ne garde que les séquences qui ont pour centre un symbole
                    posit_secu.append(posit_secu_uniq)

    return posit_secu


def comparaison_sequences(posit_sequence_reel, posit_sequence_detectee, erreur_max):

    corresp_reel_detecte = []
    for pt_reel in posit_sequence_reel:
        for pt_dect in posit_sequence_detectee:
            erreur = 0
            for i in range(len(pt_reel[1])):
                 if pt_reel[1][i] != pt_dect[1][i]:
                     erreur += 1
            if erreur <= erreur_max:
                corresp_reel_detecte.append([pt_reel[0], pt_dect[0], pt_reel[1], erreur])
    
    return corresp_reel_detecte


def tri_correspondances(sequences_corresp):
  

    #TRI PAR RAPPORT AUX POINTS REELS --> Il ne pas y avoir de doublons de points images pour un seul point réel

    corresp_reel_trie = sorted(sequences_corresp, key=lambda x: (x[0], x[3]))   #on trie par position puis par erreur
    corresp_triee_reel = []
    for cle, groupe in itertools.groupby(corresp_reel_trie, lambda x: x[0]): #on crée un groupe par position de mire réelles
        valeurs_groupe = [[x[1], x[2], x[3]] for x in groupe]
        corresp_triee_reel.append((cle, valeurs_groupe))

    sequences_pt_reel_triees = []
    for corr in corresp_triee_reel:
        nbre_appariements_possibles = len(corr[1]) #On regarde le nombre de points image qui pourraient correspondre à un seul point de la mire réelle
        if nbre_appariements_possibles == 1:
            sequences_pt_reel_triees.append([corr[0], corr[1][0][0], corr[1][0][1], corr[1][0][2]])
            #print("essai appariement", appariement)
        elif corr[1][0][2] < corr[1][1][2]: #si plusieurs appariements sont possibles, on sait que la plus petite erreur possible se trouve en première possition
            #si la première séquence a une erreur plus faible que la seconde, on prend la première, sinon on ne prend rien car on ne sait pas laquelle est la bonne
            sequences_pt_reel_triees.append([corr[0], corr[1][0][0], corr[1][0][1], corr[1][0][2]])

    #TRI PAR RAPPORT AUX POINTS REELS --> Il ne pas y avoir de doublons de points images pour un seul point réel 

    sequences_pt_image_non_triees = []
    for corr in sequences_pt_reel_triees: #on crée une liste de correspondances avec les points images non triés
        sequences_pt_image_non_triees.append([corr[1], corr[0], corr[2], corr[3]])

    corresp_img_trie = sorted(sequences_pt_image_non_triees, key=lambda x: (x[0], x[3]))   #on trie par position puis par erreur
    corresp_triee_img = []
    for cle, groupe in itertools.groupby(corresp_img_trie, lambda x: x[0]): #on crée un groupe par position de mire réelles
        valeurs_groupe = [[x[1], x[2], x[3]] for x in groupe]
        corresp_triee_img.append((cle, valeurs_groupe))

    sequences_pt_img_triees = []
    for corr in corresp_triee_img:
        nbre_appariements_possibles = len(corr[1]) #On regarde le nombre de points image qui pourraient correspondre à un seul point de la mire réelle
        if nbre_appariements_possibles == 1:
            sequences_pt_img_triees.append([corr[0], corr[1][0][0], corr[1][0][1], corr[1][0][2]])
            #print("essai appariement", appariement)
        elif corr[1][0][2] < corr[1][1][2]: #si plusieurs appariements sont possibles, on sait que la plus petite erreur possible se trouve en première possition
            #si la première séquence a une erreur plus faible que la seconde, on prend la première, sinon on ne prend rien car on ne sait pas laquelle est la bonne
            sequences_pt_img_triees.append([corr[0], corr[1][0][0], corr[1][0][1], corr[1][0][2]])

    sequences_triee_final= []
    for corr in sequences_pt_img_triees: #on crée une liste de correspondances avec les points images non triés
        sequences_triee_final.append([corr[1], corr[0], corr[2], corr[3]])

    return sequences_triee_final


global posit_sequence_reel_rot_0
global posit_sequence_reel_rot_90
global posit_sequence_reel_rot_180
global posit_sequence_reel_rot_270
global need_to_determine_real_sequences 

global corresp_reel_detect_trie_rot_0
global corresp_reel_detect_trie_rot_90
global corresp_reel_detect_trie_rot_180
global corresp_reel_detect_trie_rot_270

need_to_determine_real_sequences = True


def sequence_comparison_et_tri(posit_sequence_reel, posit_sequence_detectee, nbr_erreur_max, rot):

    global corresp_reel_detect_trie_rot_0
    global corresp_reel_detect_trie_rot_90
    global corresp_reel_detect_trie_rot_180
    global corresp_reel_detect_trie_rot_270

    corresp_reel_detect = comparaison_sequences(posit_sequence_reel, posit_sequence_detectee, nbr_erreur_max)
    corresp_reel_detect_trie = tri_correspondances(corresp_reel_detect)
    if rot ==0 : 
        corresp_reel_detect_trie_rot_0  = corresp_reel_detect_trie
    elif rot ==90 :
        corresp_reel_detect_trie_rot_90  = corresp_reel_detect_trie
    elif rot ==180 :
        corresp_reel_detect_trie_rot_180  = corresp_reel_detect_trie
    elif rot ==270 :
        corresp_reel_detect_trie_rot_270  = corresp_reel_detect_trie



def appariement_symboles_4rotations(matrice_sequence_détectée, Mire_reel, nbr_erreur_max_seq):

    global posit_sequence_reel_rot_0
    global posit_sequence_reel_rot_90
    global posit_sequence_reel_rot_180
    global posit_sequence_reel_rot_270
    global need_to_determine_real_sequences

    global corresp_reel_detect_trie_rot_0
    global corresp_reel_detect_trie_rot_90
    global corresp_reel_detect_trie_rot_180
    global corresp_reel_detect_trie_rot_270

    if(need_to_determine_real_sequences): #On ne cherche les séquences de la mire réelle qu'une seule fois car ils seront toujours identiques quelque soit la rotation de la mire
        need_to_determine_real_sequences = False
        matrice_reel_rot_90 = np.rot90(Mire_reel, k=1)
        matrice_reel_rot_180 = np.rot90(Mire_reel, k=2)
        matrice_reel_rot_270 = np.rot90(Mire_reel, k=3)

        posit_sequence_reel_rot_0 = determination_sequences_mire_reel(Mire_reel)
        posit_sequence_reel_rot_90 = determination_sequences_mire_reel(matrice_reel_rot_90)
        posit_sequence_reel_rot_180 = determination_sequences_mire_reel(matrice_reel_rot_180)  
        posit_sequence_reel_rot_270 = determination_sequences_mire_reel(matrice_reel_rot_270)  

    posit_sequence_detectee = determination_sequences_mire_detect(matrice_sequence_détectée)


    # on compare les séquences de la mire réelle avec les séquences de la mire détectée, et en même temps on enlève les doublons
    
    #cette partie se fait avec des threads, sinon c'est trop long

    """
    #partie séquentielle : 
    sequence_comparison_et_tri(posit_sequence_reel_rot_0, posit_sequence_detectee, nbr_erreur_max_seq, 0)
    sequence_comparison_et_tri(posit_sequence_reel_rot_90, posit_sequence_detectee, nbr_erreur_max_seq, 90)
    sequence_comparison_et_tri(posit_sequence_reel_rot_180, posit_sequence_detectee, nbr_erreur_max_seq, 180)
    sequence_comparison_et_tri(posit_sequence_reel_rot_270, posit_sequence_detectee, nbr_erreur_max_seq, 270)
    
    """

    #partie multithread :
    thread_rot_0 = threading.Thread(target=sequence_comparison_et_tri, args=(posit_sequence_reel_rot_0, posit_sequence_detectee, nbr_erreur_max_seq, 0))
    thread_rot_90 = threading.Thread(target=sequence_comparison_et_tri, args=(posit_sequence_reel_rot_90, posit_sequence_detectee, nbr_erreur_max_seq, 90))
    thread_rot_180 = threading.Thread(target=sequence_comparison_et_tri, args=(posit_sequence_reel_rot_180, posit_sequence_detectee, nbr_erreur_max_seq, 180))
    thread_rot_270 = threading.Thread(target=sequence_comparison_et_tri, args=(posit_sequence_reel_rot_270, posit_sequence_detectee, nbr_erreur_max_seq, 270))
    
    thread_rot_0.start()
    thread_rot_90.start()
    thread_rot_180.start()
    thread_rot_270.start()

    thread_rot_0.join()
    thread_rot_90.join()
    thread_rot_180.join()
    thread_rot_270.join()
 

    try : nbr_appariements_rot_0 = len(corresp_reel_detect_trie_rot_0)
    except  : nbr_appariements_rot_0 = 0
    try : nbr_appariements_rot_90 = len(corresp_reel_detect_trie_rot_90)
    except  : nbr_appariements_rot_90 = 0
    try : nbr_appariements_rot_180 = len(corresp_reel_detect_trie_rot_180)
    except  : nbr_appariements_rot_180 = 0
    try : nbr_appariements_rot_270 = len(corresp_reel_detect_trie_rot_270)
    except : nbr_appariements_rot_270 = 0

    moy_erreur_appariements_rot_0 = 0
    moy_erreur_appariements_rot_90 = 0
    moy_erreur_appariements_rot_180 = 0
    moy_erreur_appariements_rot_270 = 0

    if nbr_appariements_rot_0 > 0 :
        compteur_erreur = 0
        for appariements in corresp_reel_detect_trie_rot_0:
            compteur_erreur = compteur_erreur + appariements[3]
        moy_erreur_appariements_rot_0 = compteur_erreur/nbr_appariements_rot_0  
    else : 
          moy_erreur_appariements_rot_0 = 1
    if nbr_appariements_rot_90 > 0 :
        compteur_erreur = 0
        for appariements in corresp_reel_detect_trie_rot_90:
            compteur_erreur = compteur_erreur + appariements[3]
        moy_erreur_appariements_rot_90 = compteur_erreur/nbr_appariements_rot_90 
    else : 
          moy_erreur_appariements_rot_90 = 1

    if nbr_appariements_rot_180 > 0 :
        compteur_erreur = 0
        for appariements in corresp_reel_detect_trie_rot_180:
            compteur_erreur = compteur_erreur + appariements[3]
        moy_erreur_appariements_rot_180 = compteur_erreur/nbr_appariements_rot_180 
    else : 
          moy_erreur_appariements_rot_180 = 1

    if nbr_appariements_rot_270 > 0 :
        compteur_erreur = 0
        for appariements in corresp_reel_detect_trie_rot_270:
            compteur_erreur = compteur_erreur + appariements[3]
        moy_erreur_appariements_rot_270 = compteur_erreur/nbr_appariements_rot_270
    else : 
          moy_erreur_appariements_rot_270 = 1


    #recheche de l'angle optimal pour avoir le plus d'appariements possibles : 

    score_0 = nbr_appariements_rot_0 #* 2 / (moy_erreur_appariements_rot_0 + 1) # on ajoute 1 pour éviter les divisions par 0
    score_90 = nbr_appariements_rot_90 #* 2 / (moy_erreur_appariements_rot_90 + 1)
    score_180 = nbr_appariements_rot_180 #* 2 / (moy_erreur_appariements_rot_180 + 1)
    score_270 = nbr_appariements_rot_270 #* 2 / (moy_erreur_appariements_rot_270 + 1)
    score_max = max(score_0, score_90, score_180, score_270)


    if score_max == score_0:
        meilleur_appariement = corresp_reel_detect_trie_rot_0
        meilleur_angle = 0
        erreur_moy_appariement = moy_erreur_appariements_rot_0
    elif score_max == score_90:
        meilleur_appariement = corresp_reel_detect_trie_rot_90
        meilleur_angle = 90
        erreur_moy_appariement = moy_erreur_appariements_rot_90
    elif score_max == score_180:
        meilleur_appariement = corresp_reel_detect_trie_rot_180
        meilleur_angle = 180
        erreur_moy_appariement = moy_erreur_appariements_rot_180
    elif score_max == score_270:
        meilleur_appariement = corresp_reel_detect_trie_rot_270
        meilleur_angle = 270
        erreur_moy_appariement = moy_erreur_appariements_rot_270

    if meilleur_appariement is None or meilleur_angle is None or erreur_moy_appariement is None:
        meilleur_appariement = []
        meilleur_angle = 0
        erreur_moy_appariement = 0
        
    return meilleur_appariement, meilleur_angle, erreur_moy_appariement 