#Ce fichier contient des fonctions utilisées par Reco_mire.py

import cv2
import numpy as np
import math
import threading


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

def matrice_rgb_show(matrice, appariements, affich_err_appariement): #transforme la matrice numérique en matrice couleur et l'affiche
    
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
    if affich_err_appariement == True and appariements is not None and len(appariements) != 0:
        for appar in appariements : 
            coord_x = int(appar[0][0]*30 + 7)
            coord_y = int(appar[0][1]*30 + 20)
            erreur = str(appar[3])
            cv2.putText(matrice_rgb, erreur, (coord_x,coord_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

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
        
    return rot_probable

def detect_contours(img, contour_show = False):
    #détection des contours de l'image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #conversion de l'image en niveau de gris
    canny_image = cv2.Canny(gray, 50,150) #Appliquer un filtre de Canny sur l'image
    contours, _ = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #Détection des contours 
    
    if (contour_show):
        cv2.imshow('Contours', cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB)) #on affiche les contours détectés 

    return contours


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
    corresp_reel_detect_trie = sorted(sequences_corresp, key=lambda x: (x[0], x[3]))   #on trie par position puis par erreur
    #on enlève les doublons car on peut avoir plusieurs correspondances pour un point de la mire réelle

    resultat_final = []
    if len(corresp_reel_detect_trie) >1:
        resultat_final.append(corresp_reel_detect_trie[0])
        for i in range(1, len(corresp_reel_detect_trie)):

            if corresp_reel_detect_trie[i][0] != corresp_reel_detect_trie[i-1][0]:
                #print("corresp_reel_detect_trie[i][0]", corresp_reel_detect_trie[i][0])
                resultat_final.append(corresp_reel_detect_trie[i])
        
    return resultat_final



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
    #écrire le nom du thread : 
    #print("thread name = ", threading.current_thread().name)
    #print("Nombre de threads actifs = ", threading.active_count())
    #print("rot = ", rot)
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


    #print("nbr_appariements_rot_0 : ", nbr_appariements_rot_0)
    #print("nbr_appariements_rot_90 : ", nbr_appariements_rot_90)
    #print("nbr_appariements_rot_180 : ", nbr_appariements_rot_180)
    #print("nbr_appariements_rot_270 : ", nbr_appariements_rot_270)
    #print("############################################")
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