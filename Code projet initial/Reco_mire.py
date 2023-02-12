#Camera Autofocus project : FIP-EII-2A Telecom Physique Strasbourg 2022-2023

#Authors :
#   - DOTTE Antoine
#   - JENNY Thibaud
#   - FISCHER Arnaud

#Date : 01/02/2023

#Description : This code is used to detect the symbols of the camera autofocus test chart and to display them in a matrix


import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

cap = cv2.VideoCapture(0)

ratio_list = [] #ratio des symboles

#####Variables globales##############################################################################################################
mat_dim_mire = np.zeros((15,15)) #Dimensions de la matrice

nbr_row_matrice = mat_dim_mire.shape[1] 
nbr_col_matrice = mat_dim_mire.shape[0]

matrice_circulaire_mire_symbole = [] #liste qui contiendra les matrices symboles pour moyennage des symboles
num_pos_matrice_RGB_moy = 0 #variable qui permet de savoir quelle matrice de la liste matrice_RGB_moy on doit remplir
nbr_moyennage_matrice_symbole = 30 #nombre de moyennage de la matrice des symboles : 30 c'est bien
real_symbol_length = 0.4 #longueur d'un symbole réel en cm

for i in range(nbr_moyennage_matrice_symbole):
    matrice_circulaire_mire_symbole.append(mat_dim_mire) #on remplit la liste avec des matrices de 15*15

Mire_reel_ancien = [[1,2,2,2,1,2,1,2,2,2,1,2,2,2,1],#correspond à la mire réelle (normalement)
            [2,2,2,1,1,1,2,1,1,2,2,1,2,2,1], 
            [2,1,2,2,1,2,2,2,2,2,2,2,2,2,1],
            [1,2,1,2,2,2,1,2,2,2,2,2,1,2,2],
            [1,1,2,1,2,1,2,2,2,1,1,2,1,2,2],
            [2,1,2,2,2,1,2,1,2,1,2,2,2,1,2],
            [2,2,1,1,2,2,2,1,2,2,1,2,1,1,2],
            [1,2,2,2,1,2,1,1,1,2,2,1,2,1,1],
            [2,1,2,2,1,2,2,2,2,2,2,1,2,2,1],
            [2,2,2,1,1,1,2,1,2,2,2,2,1,2,2],
            [1,2,2,2,2,1,1,2,2,2,2,1,2,2,1],
            [2,2,2,1,2,1,2,1,2,2,1,1,1,2,1],
            [2,2,1,2,1,2,2,2,2,2,1,1,2,2,2],
            [2,1,2,1,1,2,1,1,2,1,2,2,1,2,2],
            [2,2,2,2,1,2,1,2,1,1,2,1,2,2,2]]


Mire_reel =  [[1, 3, 3, 2, 1, 3, 1, 2, 3, 3, 1, 3, 2, 2, 1], #si = 1, trait. Si = 2, ronds, si 3 = cercle
                [3, 2, 2, 1, 1, 1, 2, 1, 1, 3, 2, 1, 3, 3, 1], 
                [2, 1, 2, 2, 1, 2, 2, 2, 3, 3, 3, 2 ,3, 3, 1], 
                [1, 3, 1, 3, 3, 3, 1, 3, 2, 2, 2, 3, 1, 2, 3], 
                [1, 1, 3, 1, 3, 1, 2, 3, 2, 1, 1, 3, 1, 2, 3], 
                [2, 1, 3, 3, 2, 1, 2, 1, 2, 1, 3, 2, 2, 1, 2], 
                [2, 3, 1, 1, 2, 3, 3, 1, 3, 3, 1, 2, 1, 1, 2],
                [1, 2, 2, 3, 1, 2, 1, 1, 1, 3, 2, 1, 2, 1, 1], 
                [3, 1, 2, 2, 1, 3, 2, 3, 2, 2 ,3, 1, 2, 3 ,1], 
                [3, 3, 3, 1, 1, 1, 3, 1, 2, 3, 3, 3, 1 ,3 ,2], 
                [1, 3, 3, 2, 2, 1, 1 ,3, 3, 2, 3, 1, 3, 2, 1], 
                [3, 3, 2, 1, 3, 1, 3, 1, 3, 3, 1, 1, 1, 3, 1], 
                [2, 2, 1, 2, 1, 3, 2, 2, 2, 3, 1, 1 ,2, 3, 3], 
                [3, 1, 2, 1, 1, 3, 1, 1, 2, 1, 2, 3, 1, 2, 2], 
                [3, 2, 3, 3, 1, 2, 1, 3, 1, 1, 3, 1, 2, 3, 3]] #correspond à la mire réelle (normalement)


mode =  1 #si 0, on garde les mêmes coordonnées pour la mire. Si 1, on met à jours en temps réel : change en appuyant sur la touche "q"
box_if_freeze = [[0, 0], [0, 0], [0, 0], [0, 0]] #coordonnées de la boite englobante de la mire
x_pos_mire_if_freeze = 0
y_pos_mire_if_freeze = 0
angle_mire_if_freeze = 0

#####Fonctions#######################################################################################################################


def angle_inclinaison_rectangle(pts): #fonction qui permet de trouver l'angle d'inclinaison d'un rectangle en passant par le calcul de son vecteur normal
    #MARCHE PAS TRES BIEN
    #points = np.array(pts, dtype="int32")

    [vx,vy,x,y] = cv2.fitLine(pts,cv2.DIST_L2,0,0.01,0.01) #trouve de vecteur normal

    inclination_x_deg = math.degrees(math.atan2(vx, 1)) #calcule l'angle d'inclinaison selon X en degrés
    inclination_y_deg = math.degrees(math.atan2(vy, 1)) #calcule l'angle d'inclinaison selon Y en degrés

    return inclination_x_deg, inclination_y_deg


def moyennage_matrice_symbole(matrice): #fonction qui permet de faire un moyennage des symboles détectés sur plusieurs images
    
    global num_pos_matrice_RGB_moy
    global matrice_circulaire_mire_symbole
    global nbr_row_matrice
    global nbr_col_matrice

    if num_pos_matrice_RGB_moy < len(matrice_circulaire_mire_symbole)-1:
        num_pos_matrice_RGB_moy = num_pos_matrice_RGB_moy + 1
    else:
        num_pos_matrice_RGB_moy = 0
    matrice_circulaire_mire_symbole[num_pos_matrice_RGB_moy] = matrice

    matrice_moyennee = np.zeros((nbr_row_matrice, nbr_col_matrice), dtype=np.uint8)
    for i in range(nbr_row_matrice):
        for j in range(nbr_col_matrice):
            moy_pixel = []
            for k in range(len(matrice_circulaire_mire_symbole)):
                moy_pixel.append(matrice_circulaire_mire_symbole[k][i, j])
            moy_pixel_sans_0 = list(filter(lambda x: x != 0, moy_pixel)) #on enlève les 0 de la liste
            if len(moy_pixel_sans_0) == 0: #si la liste est vide, on met 0 dans la matrice moyennee
                matrice_moyennee[i, j] = 0
            else:
                nbr_1 = 0
                nbr_2 = 0
                nbr_3 = 0
                for l in range(len(moy_pixel_sans_0)):
                    if moy_pixel_sans_0[l] == 1 :
                        nbr_1 = nbr_1 + 1
                    elif moy_pixel_sans_0[l] == 2:
                        nbr_2 = nbr_2 + 1
                    elif moy_pixel_sans_0[l] == 3:
                        nbr_3 = nbr_3 + 1

                if max(nbr_1, nbr_2, nbr_3) == nbr_1:
                    matrice_moyennee[i, j] = 1
                elif max(nbr_1, nbr_2, nbr_3) == nbr_2:
                    matrice_moyennee[i, j] = 2
                elif max(nbr_1, nbr_2, nbr_3) == nbr_3:
                    matrice_moyennee[i, j] = 3
    return matrice_moyennee

def pourcent_symbole_detected(matrice): #fonction qui permet de calculer le pourcentage de symbole détecté

    global nbr_row_matrice
    global nbr_col_matrice

    symbole_found = np.count_nonzero(np.array(matrice))
    pourcent_symb_detect = round(100 * symbole_found/(nbr_row_matrice*nbr_col_matrice), 3)
    return pourcent_symb_detect

def matrice_rgb_show(matrice): #transforme la matrice numérique en matrice couleur et l'affiche
    
    global nbr_row_matrice
    global nbr_col_matrice

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



def  matrice_normalisation(matrice): #fonction qui permet de normaliser la matrice pour placer les symboles dans une matrice de taille nbr_row_matrice*nbr_col_matrice 

    global nbr_row_matrice
    global nbr_col_matrice

    normalised_points_x = [point[0] for point in matrice]
    normalised_points_y = [point[1] for point in matrice]

    min_x = np.min(normalised_points_x)
    min_y = np.min(normalised_points_y)
    max_x = np.max(normalised_points_x)
    max_y = np.max(normalised_points_y)

    normalised_points_x = ((normalised_points_x - min_x)/(max_x - min_x)) * (nbr_row_matrice-1) #on normalise les coordonnées des points pour qu'ils soient compris entre 0 et 15
    normalised_points_y = ((normalised_points_y - min_y)/(max_y - min_y)) * (nbr_col_matrice-1) ##on normalise les coordonnées des points pour qu'ils soient compris entre 0 et 15

    normalised_points = np.transpose(np.array([normalised_points_x, normalised_points_y]))
    return normalised_points


def matrice_rotation(angle, mean_x, mean_y, matrice): #fonction qui permet de tourner la matrice de symbole pour qu'elle soit bien orientée en fonction de l'angle de la mire détecté
#MARCHE
    angle_rad = math.radians(-angle) #on convertit l'angle en radians
    rotation_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]]) #on crée la matrice de rotation
    centre_matrix = np.array([mean_x, mean_y])
    rotated_matrice = []
    for point in matrice:
        points = point - centre_matrix
        rotated_point = np.dot(rotation_matrix, points)
        rotated_matrice.append(rotated_point)
    return rotated_matrice

def variance_of_image_blur_laplacian(image): #Calcule la valeur de flou de l'image avec le Laplacien
	return round(cv2.Laplacian(image, cv2.CV_64F).var(), 2)

def variance_of_image_blur_sobel(image): #Calcule la valeur de flou de l'image avec le Sobel --> méthode de la dérivée seconde
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5) #calcul du gradient de Sobel selon l'axe x
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5) #calcul du gradient de Sobel selon l'axe y
    
    return cv2.norm(sobelx, sobely, cv2.NORM_L2) #calcul de la norme des deux gradients

def variance_of_image_blur_Canny(image):#Calcule la valeur de flou de l'image avec le Canny
	return round(cv2.Canny(image, 100, 200).var(), 1)

def variance_of_image_blur_entropy(image):#Calcule la valeur de flou de l'image par méthode statistique avec l'entropie : dispersion des pixels de l'image
    hist = np.histogram(image, bins=256, range=(0, 255))[0]
    hist = hist / hist.sum()
    return -np.sum(hist * np.log2(hist + 1e-7))

def variance_of_image_blur_corner_counter(image): #Calcule la valeur de flou de l'image avec le nombre de coins détectés
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(image, None)
    return len(kp)

def variance_of_image_blur_picture_variability(image): #Calcule la valeur de flou de l'image avec la variabilité de l'image
    mean , stddev = cv2.meanStdDev(gray)
    return mean[0][0]

def symbole_identique_2_matrices_V3(matrice1, matrice2): #calcul de l'erreur quadratique moyenne entre deux matrices
    h, w = np.array(matrice1).shape
    diff = matrice1 - matrice2
    err = np.sum(diff**2)
    mse = err/float((h*w))
    return round(mse, 5)



def symbole_identique_2_matrices_V4(matrice1, matrice2): #calcul de l'erreur entre 2 matrices et ne pénalise pas si un symbole n'est pas détecté
    juste = 0
    num_elements = 0
    penalité = 5

    global nbr_row_matrice
    global nbr_col_matrice
    for i in range(nbr_row_matrice):
        for j in range(nbr_col_matrice):
            if matrice2[i][j] !=0: #si le symbole n'est pas détecté, on passe
                if matrice1[i][j] == matrice2[i][j]:
                    juste = juste + 1

    pourcent_erreur = 100 * (1-(juste / (nbr_row_matrice*nbr_col_matrice)))       
    return np.round(pourcent_erreur, 5)


def comparison_mire(real_mire, matrice_symb): #fonction globale pour comparer la mire originelle avec la mire potentielle trouvée
    #MARCHE mais à améliorer

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




def perspective_mire(image, box): #fonction pour transformer l'image de mire en perspective --> Sert à rien en tant que tel, mais stylé

    pts1 = np.float32([box[0], box[1], box[3], box[2]])
    pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])        

    matrix = cv2.getPerspectiveTransform(pts1, pts2 )
    perspective = cv2.warpPerspective(image, matrix, (300, 300))
    cv2.imshow('perspective', perspective)

###################################################################################################################################################################################################################################################



while True: 

    nbr_circle_detected = 0 #variable d'incrémetation
    long = 0 #longueur du symbole, sera utilisé pour la normalisation de la mire
    matrice_symb = np.zeros((15, 15), dtype=int) #on crée une matrice de 15x15 --> taille de la mire qui contiendra les symboles détectés
    matrice_symbole_x = [] #contient les coordonnées x des symboles de la mire trouvés
    matrice_symbole_y = [] #contient les coordonnées y des symboles de la mire trouvés
    matrice_symbole_type = [] #contient le type de symbole (1 = trait, 2 = cercle) trouvés
    matrice_symbole_rayon = [] #contient le rayon des cercles trouvés pour pouvoir faire la distinction des disques ou des ronds pleins

    mean_x = 0 #moyenne des coordonnées x des symboles de la mire trouvés
    mean_y = 0 #moyenne des coordonnées y des symboles de la mire trouvés
    angle = 0 #angle de rotation de la mire
    #LIGNE CODE DEBUT************************************************************************************************************************************************************************************

    ret, frame = cap.read() #prendre une image

    #creation des images de travail
    frame_orig  = frame.copy() #copier l'image pour pouvoir la réutiliser pour la mire avec le masque
    frame_symb_only = cv2.cvtColor(np.full(frame.shape[:2], 0, dtype=np.uint8), cv2.COLOR_GRAY2RGB) #création d'une image noire sur laquelle sera intercallée les symboles détectés
    mire = np.full(frame.shape[:2], 0, dtype=np.uint8) #création d'une image noire sauf à l'endroit de la mire pour le calcul de la variance de l'image

    #détection des contours
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convertir en noir et blanc
    canny_image = cv2.Canny(frame, 50,150) #Appliquer un filtre de Canny pour détecter les contours
    contours, _ = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #trouver les contours de l'image avec le filtre de Canny
    canny_image_cont = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB) #convertir en couleur pour pouvoir afficher les contours en couleur --> frame avec contours
    cv2.imshow('Contours', canny_image_cont) #on affiche l'image avec les contours
    
    #DETECTION DES CERCLES ************************************************************************************************************************************************************************************
    for cnt in contours:    
        if cv2.contourArea(cnt) > 5: # Vérifier si le contour est suffisamment grand pour être considéré comme une forme

            rect = cv2.minAreaRect(cnt) # Obtenir le rectangle minimum qui englobe les points de contour
            (x, y), (w, h), angle = rect
            ratio = h/w
            #ratio_list.append(round(h/w, 4))
            if 0.999 < ratio <1.001 :  #détection des cercles : on considère que c'est des carrés parfaits donc ratio = 1 (plus facile à détecter que des cercles eux mêmes)              
                long = max(w, h) + long #longueur du symbole
                matrice_symbole_x.append(int(x)) #ajout de la coordonnée x du symbole dans la matrice des coordonnées x des symboles
                matrice_symbole_y.append(int(y)) #ajout de la coordonnée y du symbole dans la matrice des coordonnées y des symboles
                matrice_symbole_type.append(2) #ajout du type du symbole dans la matrice des types des symboles
                matrice_symbole_rayon.append(int(min(w/2,h/2))) #ajout du rayon du cercle dans la matrice des rayons des cercles
                cv2.circle(frame_symb_only, (int(x), int(y)), int(min(w/2,h/2)), (255,0,0), 2) #dessiner un cercle sur l'image des symboles détectés              
                nbr_circle_detected = nbr_circle_detected+1
    if nbr_circle_detected>0:
        long_trait_max = long/nbr_circle_detected #calcul de la longueur moyenne des traits de la mire
    else:
        long_trait_max = 0
    

    #DETECTION DES TRAITS ************************************************************************************************************************************************************************************
    for cnt in contours:     
        # Vérifier si le contour est suffisamment grand pour être considéré comme une forme
        if cv2.contourArea(cnt) > 4: # Obtenir le rectangle minimum qui englobe les points de contour
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            ratio = h/w
            #ratio_list.append(round(h/w, 4))
            #if max(w, h) < long_trait_max*1.6 and (0.10 < ratio <0.5 or 2 < ratio <10): #les rectangles sont environ 5fois plus grand en longueur que el largeur : ratio = env 3.5 ou env 0.28
            if max(w, h) < long_trait_max*1.6 and (0.2 < ratio <0.5 or 1.8 < ratio <5):
                box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(frame_symb_only,[box],0,(0,0,255),2) #dessiner un rectangle sur l'image des symboles détectés

                matrice_symbole_x.append(int(x)) #ajout de la coordonnée x du symbole dans la matrice des coordonnées x des symboles
                matrice_symbole_y.append(int(y)) #ajout de la coordonnée y du symbole dans la matrice des coordonnées y des symboles
                matrice_symbole_type.append(1) #ajout du type du symbole dans la matrice des types des symboles
                matrice_symbole_rayon.append(0)
                nbr_circle_detected=nbr_circle_detected+1 

    #ENLEVER LES SYMBOLES FAUX POSITIFS ************************************************************************************************************************************************************************************
    
    points_et_type_et_rayon = np.array(np.column_stack((matrice_symbole_x, matrice_symbole_y, matrice_symbole_type, matrice_symbole_rayon)), dtype = np.int32)
    points = np.array(np.column_stack((matrice_symbole_x, matrice_symbole_y, )), dtype = np.int32)

    treshold = long_trait_max * 23 #distance pour que les points soient considérés comme des vraies symboles : cercle de rayon de 20 fois la longueur moyenne des traits centré sur le centre de gravité de la mire

    if points.size != 0: #si il y a des symboles détectés
        mean = np.mean(points, axis=0) #calcul de la moyenne des coordonnées des symboles
        distance = np.linalg.norm(points - mean, axis=1) #calcul de la distance entre chaque point et la moyenne
        good_points_et_type_et_rayon = points_et_type_et_rayon[distance < treshold] #on garde les points qui sont à moins de la distance treshold de la moyenne
        good_points = points[distance < treshold] 
    
        if len(good_points_et_type_et_rayon) >=10: #si il y a au moins 10 symboles proches, on considère qu'on a trouvé la mire
            hull = cv2.convexHull(np.array(good_points))
            
            #dessiner le rectangle minimum qui englobe les points de contour
            rect = cv2.minAreaRect(good_points) 


            width_rectangle_mire, height_rectangle_mire = rect[1]
           
            box = np.int0(cv2.boxPoints(rect))
           # incli_X, incli_Y = angle_inclinaison_rectangle(box)
            #print("incli_X = ", int(incli_X), "incli_Y = ", int(incli_Y))

    
            
        #TRI DES SYMBOLES & POSITION DE LA MIRE **************************************************************************************************************************************************************
            moments = cv2.moments(box)

            if rect is not None and moments["m00"] != 0 :
                #calcul des moments de l'image pour trouver son centre de gravité
                mean_x = int(moments["m10"] / moments["m00"]) 
                mean_y = int(moments["m01"] / moments["m00"])   

                
                angle = int(rect[-1]) #angle du rectangle englobant
                if angle == 90:
                    angle = 0
                arrow_length = 100
                end_x = int(mean_x + arrow_length * np.cos(np.deg2rad(angle)))
                end_y = int(mean_y + arrow_length * np.sin(np.deg2rad(angle)))   
            cv2.circle(frame_symb_only, (mean_x, mean_y), int(treshold), (0,100, 0), 2) #dessiner un cercle sur l'image des symboles détectés 
        #CREATION DE LA MATRICE SYMBOLE CORRESPONDANT A LA MIRE ET AUX BONS SYMBOLES*******************************************************************************************************************************

    rotation_probable = 0
    try: good_points_et_type_et_rayon
    except NameError: 
        good_points_et_type_et_rayon = None
        matrice_symb_moyenne = None

    if good_points_et_type_et_rayon is not None: #si il y a des symboles 
        if len(good_points_et_type_et_rayon) >=15: #si il y a suffisament de symboles
            if box is not None: #si la mire est détectée

                rotated_points = matrice_rotation(angle, mean_x, mean_y, good_points) #rotation de la matrice contenant les coordonnées des symboles
                normalised_points = matrice_normalisation(rotated_points) #normalisation de la matrice contenant les coordonnées des symboles

            i=0
            #calcul du rayon moyen des cercles détectés. Si le cercle est plus petit que la moyenne, alors on a un cercle et non un rond
            #qqmean_rayon = np.mean(good_points_et_type_et_rayon[:, 3])
            mean_rayon= np.mean(list(filter(lambda x: x != 0, good_points_et_type_et_rayon[:, 3]))) #on enlève les 0 de la liste
            for x, y in normalised_points: #on met les symboles dans la nouvelle matrice normalisée et rotationnée
                if matrice_symb[int(np.around(x))][int(np.around(y))] == 2 and good_points_et_type_et_rayon[i, 3] == 2 : #Si on a 2 symboles rond, alors c'est un cercle
                    matrice_symb[int(np.around(x))][int(np.around(y))] = 3 #on met un cercle

                elif good_points_et_type_et_rayon[i, 3] == 0 : #On a un trait
                    matrice_symb[int(np.around(x))][int(np.around(y))] = 1 #on met un trait
                else: #Soit un trait soit un rond
                    
                    if matrice_symb[int(np.around(x))][int(np.around(y))] ==3 : #Si déjà un cercle détecté, on passe car on veut pas le remplacer par un rond
                        pass
                    elif matrice_symb[int(np.around(x))][int(np.around(y))] == 0 : #Si pas de symbole,

                        if good_points_et_type_et_rayon[i, 3] < mean_rayon*0.98 : #Si le rayon du symbole est plus petit que la moyenne, alors on a un cercle
                            matrice_symb[int(np.around(x))][int(np.around(y))] = 3 #on met un cercle
                        else: #Sinon on a un rond
                            matrice_symb[int(np.around(x))][int(np.around(y))] = 2 #on met un rond
                i+=1 
        matrice_symb_moyenne = moyennage_matrice_symbole(matrice_symb) #retourne la matrice symbole moyennée sur 5 images --> permet de mieux connaitre la matrice symbole
    
    
    if np.count_nonzero(matrice_symb_moyenne)>10 : #si il y a au moins 5 symboles
        rotation_probable = comparison_mire(Mire_reel, matrice_symb_moyenne)
        #perspective_mire(frame, box) #optionnel, affiche la perspective de la mire
    
    if  np.count_nonzero(matrice_symb_moyenne)>15 or mode ==2: #si il y a au moins 5 symboles ou si on est en mode 2 --> freeze de la position de la mire
        cv2.putText(frame, "{}{}".format(" Angle de rotation probable de la mire : angle = ", rotation_probable + int(angle)), (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 
        cv2.circle(frame, (mean_x, mean_y), 10, (255, 255, 0), 2)
        cv2.putText(frame, "{}{}{}{}".format(" Position du centre de gravite de la mire : X=", mean_x," ; Y=", mean_y), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  
        cv2.putText(frame, "{}{}".format(" Vecteur de rotation du rectangle englobant : angle = ", int(angle)), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)         
        cv2.arrowedLine(frame, (mean_x, mean_y), (end_x, end_y), (255, 255, 0), 1)

        angle_total = rotation_probable + angle #angle du rectangle englobant
        arrow_length = 150
        end_x_angle_tot = int(mean_x + arrow_length * np.cos(np.deg2rad(angle_total)))
        end_y_angle_tot = int(mean_y + arrow_length * np.sin(np.deg2rad(angle_total)))  

        cv2.arrowedLine(frame, (mean_x, mean_y), (end_x_angle_tot, end_y_angle_tot), (50, 90, 255), 1)
        cv2.polylines(frame, [hull], True, (0,255,0), 2) # dessiner le polygone convexe correspondant à la mire
        cv2.polylines(frame_symb_only, [hull], True, (0,255,0), 2) # dessiner le polygone convexe correspondant à la mire
        cv2.drawContours(frame, [box], 0, (0,0,255), 2) 
        #MASQUE AVEC QUE LA MIRE ET CALCUL DE FLOU*******************************************************************************************************************************************************************************
    mode_name = ""
    if cv2.waitKey(1) == ord('r'): #on change de mode
        if mode ==0 : 
            mode = 1
        else:
            mode = 0
    if mode ==1: #
        mode_name = " Mise a jour des contours"
        try: box_if_freeze = box
        except NameError:
            pass
        x_pos_mire_if_freeze = mean_x
        y_pos_mire_if_freeze = mean_y
        angle_mire_if_freeze = angle
    else:
        mode_name = " Freeze des contours"

    if np.count_nonzero(matrice_symb_moyenne)>15 or mode == 0 : #si il y a au moins 15 symboles
        cv2.fillConvexPoly(mire, box_if_freeze,255) #on remplit la mire avec du blanc
        mire = cv2.bitwise_and(frame_orig, frame_orig, mask=mire) #on applique la mire sur l'image de départ

        gray_mire = cv2.cvtColor(mire, cv2.COLOR_BGR2GRAY)
        fm_sobel = int(variance_of_image_blur_sobel(gray_mire))
        fm_laplacian = int(variance_of_image_blur_laplacian(gray_mire))
        fm_canny = int(variance_of_image_blur_Canny(gray_mire))
        fm_entropie = round(variance_of_image_blur_entropy(gray_mire), 4)
        fm_corner_counter = variance_of_image_blur_corner_counter(gray_mire)
        fm_picture_variance = round(variance_of_image_blur_picture_variability(gray_mire), 3)

        cv2.putText(mire, "{}{}".format(" Sobel focus value : ", fm_sobel), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(mire, "{}{}".format(" Laplacian focus value : ", fm_laplacian), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(mire, "{}{}".format(" Canny focus value : ", fm_canny), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.putText(mire, "{}{}".format(" Entropy focus value : ", fm_entropie), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(mire, "{}{}".format(" Number of corner detected: ", fm_corner_counter), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(mire, "{}{}".format(" Picture variance : ", fm_picture_variance), (0, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (122, 255, 122), 1, cv2.LINE_AA)

        cv2.putText(mire, "{}".format(mode_name), (0, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        if mode==0:
                cv2.putText(mire, "{}{}{}{}{}{}".format(" Position mire : X=", x_pos_mire_if_freeze," ; Y=", y_pos_mire_if_freeze," ; angle =", angle_mire_if_freeze), (0, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  
 


    #LIGNE CODE FIN************************************************************************************************************************************************************************************ 
 
    if matrice_symb_moyenne is not None: #si la matrice des symboles n'est pas vide
        matrice_rgb_show(matrice_symb_moyenne) # transforme la matrice symbole en image RGB et l'affiche

    cv2.imshow('frame', frame) #on affiche l'image de base
    cv2.imshow('frame_symb_only', frame_symb_only) #on affiche l'image avec les symboles
    cv2.imshow('mire', mire) #on affiche la mire

    if cv2.waitKey(1) == ord('q'): #waitkey(0) --> only one frame, waitkey(1) --> video 
        break #si q est appuyé, on quitte la boucle

cap.release()
cv2.destroyAllWindows()


#pour afficher la répartition des ratios
#plt.hist(ratio_list, bins=10000)
#plt.xlim(0, 5)
#plt.grid()
#qplt.show()