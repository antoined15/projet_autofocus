#Camera Autofocus project : FIP-EII-2A Telecom Physique Strasbourg 2022-2023

#Authors :
#   - DOTTE Antoine
#   - JENNY Thibaud
#   - FISCHER Arnaud

#Date : 01/02/2023



import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

cap = cv2.VideoCapture(0)


def  matrice_normalisation(matrice, X_value, Y_value): #fonction qui permet de normaliser la matrice pour placer les symboles dans une matrice de taille X_value_wanted*Y_value_wanted 
#MARCHE
    normalised_points_x = [point[0] for point in matrice]
    normalised_points_y = [point[1] for point in matrice]

    min_x = np.min(normalised_points_x)
    min_y = np.min(normalised_points_y)
    max_x = np.max(normalised_points_x)
    max_y = np.max(normalised_points_y)

    normalised_points_x = ((normalised_points_x - min_x)/(max_x - min_x)) * (X_value-1) #on normalise les coordonnées des points pour qu'ils soient compris entre 0 et 15
    normalised_points_y = ((normalised_points_y - min_y)/(max_y - min_y)) * (Y_value-1) ##on normalise les coordonnées des points pour qu'ils soient compris entre 0 et 15

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


def matrice_found_rgb_show(matrice, X_value, Y_value): #transforme la matrice numérique en matrice couleur
#MARCHE
    matrice_rgb = np.zeros((X_value, Y_value, 3), dtype=np.uint8)
    colors = [(0, 0, 0), (0, 0, 255), (255,0 , 0)]
    for i in range(X_value):
        for j in range(Y_value):
            try : 
                matrice_rgb[j, i] = colors[matrice[i, j]]
            except:
                matrice_rgb[i, j] = (0, 0, 0)
    matrice_rgb = cv2.resize(matrice_rgb, (400, 400), cv2.INTER_LINEAR)
    cv2.imshow('matrice des symboles détectés', matrice_rgb) #on affiche la matrice de symbole


def variance_of_image_blur_laplacian(image): #MARCHE
	# compute the Laplacian of the image and then return the focus
	return round(cv2.Laplacian(image, cv2.CV_64F).var(), 2)

def variance_of_image_blur_sobel(image):#MARCHE
	# compute the Sobel of the image and then return the focus
	return round(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3 ).var(), 1)

def variance_of_image_blur_Canny(image):#MARCHE
	# compute the Canny of the image and then return the focus
	return round(cv2.Canny(image, 100, 200).var(), 1)

def symbole_identique_2_matrices(matrice1, matrice2): #calcule le nombre de symboles identiques entre deux matrices pour la même position
    #MARCHE mais à améliorer, pas pertient
    symbole_validated = 0
    for i in range(15):
        for j in range(15):
            if matrice1[i][j]==matrice2[i][j]:
                symbole_validated = symbole_validated + 1
    return round(100 * symbole_validated/225, 3)

def symbole_identique_2_matrices_V2(matrice1, matrice2): #calcule le nombre de symboles identiques entre deux matrices pour la même position
    #MARCHE mais à améliorer, pas pertient
    #matrice de taille 15x15
    matrice_mire_avec_zéros = np.zeros((15,15))
    symbole_validated = 0
    #1] on enlève à la matrice real les manques de la matrice potentielle
    #print("matrice1avant : ", matrice1)
    for i in range(15):
        for j in range(15):
            if matrice2[i][j] != 0:
                matrice_mire_avec_zéros[i][j]= matrice1[i][j] #on met la valeur de la matrice1 dans la matrice_mire_avec_zéros


    print("matrice mire avec zéros : ", matrice_mire_avec_zéros)
    print("matrice2 : ", matrice2)
    #2] On regarde ligne par ligne si c'est identique
    for i in range(15):
        if np.array_equal(matrice_mire_avec_zéros[i], matrice2[i]):
            symbole_validated = symbole_validated + 1
    return round(100 * (symbole_validated)/15, 3)


def symbole_identique_2_matrices_V3(matrice1, matrice2): #calcul de l'erreur quadratique moyenne entre deux matrices, mais marche pas mieux que les autres
    #MARCHE mais à améliorer, pas pertient
    h, w = np.array(matrice1).shape
    diff = matrice1 - matrice2
    err = np.sum(diff**2)
    mse = err/float((h*w))
    return round(mse, 4)


def comparison_mire(real_mire, matrice_symb): #fonction globale pour comparer la mire originelle avec la mire potentielle trouvée
    #MARCHE mais à améliorer
    symbole_found = np.count_nonzero(np.array(matrice_symb))
    pourcent_symb_detect = round(100 * symbole_found/225, 3)
    print("Pourcentage de symboles détectés : ", pourcent_symb_detect, "%")


    matrice_symb_rot_90 = np.rot90(matrice_symb, k=1)
    matrice_symb_rot_180 = np.rot90(matrice_symb, k=2)
    matrice_symb_rot_270 = np.rot90(matrice_symb, k=3)

    pourcent_symb_valide_0 = symbole_identique_2_matrices_V3(real_mire, matrice_symb)
    pourcent_symb_valide_90 = symbole_identique_2_matrices_V3(real_mire, matrice_symb_rot_90)
    pourcent_symb_valide_180 = symbole_identique_2_matrices_V3(real_mire, matrice_symb_rot_180)
    pourcent_symb_valide_270 = symbole_identique_2_matrices_V3(real_mire, matrice_symb_rot_270)

    print("Mean Square Error : lower the value, better the shape")
    print("r=0° : ",pourcent_symb_valide_0,)
    print("r=90° : ",pourcent_symb_valide_90)
    print("r=180° : ",pourcent_symb_valide_180)
    print("=270° : ",pourcent_symb_valide_270)

    max_symb_valid = min(pourcent_symb_valide_0, pourcent_symb_valide_90, pourcent_symb_valide_180, pourcent_symb_valide_270)
    if max_symb_valid == pourcent_symb_valide_0:
        rot_probable = 0
    elif max_symb_valid == pourcent_symb_valide_90:
        rot_probable = 90
    elif max_symb_valid == pourcent_symb_valide_180:
        rot_probable = 180
    else:
        rot_probable = 270
    print("Rotation de la mire probable :", rot_probable, "°")
    print("----------------")
    return rot_probable

 #SI LIGNES OU COLONNES VIDES DANS LA MATRICE TROUVEE,ALORS CA VEUT DIRE QUE LA MIRE N'EST PAS COMPLETE--> DECALER LES MOTIFS EN CONSEQUENCE --> REFAIRE LA NORMALISATION EN FCT DE LA LONG ET LARGEUR DU RECTANGLE ENGLOBANT
 #SI MIRE =0, ALORS ON SAIS APS
 #REMPLACER LES RONDS OU CERCLES PAR DES VIDES --> plus de points de comparaison entre les différentes matrices de rotation
 #ESSAYER DE TROUVER LA MIRE MAIS SEULEMENT AVEC LES RECTANGLES ? OU EN FAISANT DES MATRICES 4*4 ET VOIR SI C'EST IDENTIQUE
#faire une image avec opencv pour voir ce que la matrice renvoie avec des ronds de couleur par exemple


###################################################################################################################################################################################################################################################

mode =  1 #si 0, on garde les mêmes coordonnées pour la mire. Si 1, on met à jours en temps réel : change en appuyant sur la touche "q"
box_2 = [[0, 0], [0, 0], [0, 0], [0, 0]] #coordonnées de la boite englobante de la mire


while True: 

    mire_reel = [[1,2,2,2,1,2,1,2,2,2,1,2,2,2,1],#correspond à la mire réelle (normalement)
            [2,2,2,1,1,1,2,1,1,2,2,1,2,2,1], #si = 1, trait. Si = 2, ronds ou point
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

    X_value_wanted = 15 #on veut une matrice de 15x15
    Y_value_wanted = 15
    n = 0 #variable d'incrémetation
    long = 0 #longueur du symbole, sera utilisé pour la normalisation de la mire
    matrice_symb = np.zeros((15, 15), dtype=int) #on crée une matrice de 15x15 --> taille de la mire qui contiendra les symboles détectés
    matrice_symbole_x = [] #contient les coordonnées x des symboles de la mire trouvés
    matrice_symbole_y = [] #contient les coordonnées y des symboles de la mire trouvés
    matrice_symbole_type = [] #contient le type de symbole (1 = trait, 2 = cercle) trouvés

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
            if ratio == 1:  #détection des cercles : on considère que c'est des carrés parfaits donc ratio = 1 (plus facile à détecter que des cercles eux mêmes)
                box = np.int0(cv2.boxPoints(rect))
                
                long = max(w, h) + long #longueur du symbole
                matrice_symbole_x.append(int(x)) #ajout de la coordonnée x du symbole dans la matrice des coordonnées x des symboles
                matrice_symbole_y.append(int(y)) #ajout de la coordonnée y du symbole dans la matrice des coordonnées y des symboles
                matrice_symbole_type.append(2) #ajout du type du symbole dans la matrice des types des symboles
                
                cv2.circle(frame_symb_only, (int(x), int(y)), int(min(w/2,h/2)), (255,0,0), 2) #dessiner un cercle sur l'image des symboles détectés              
                n = n+1
    if n>0:
        long_trait_max = long/n #calcul de la longueur moyenne des traits de la mire
    else:
        long_trait_max = 0
    

    #DETECTION DES TRAITS ************************************************************************************************************************************************************************************
    for cnt in contours:     
        # Vérifier si le contour est suffisamment grand pour être considéré comme une forme
        if cv2.contourArea(cnt) > 5: # Obtenir le rectangle minimum qui englobe les points de contour
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect
            ratio = h/w
            if max(w, h) < long_trait_max*1.5 and (0.10 < ratio <0.5 or 2 < ratio <10): #les rectangles sont environ 5fois plus grand en longueur que el largeur : ratio = env 3.5 ou env 0.28
                box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(frame_symb_only,[box],0,(0,0,255),2) #dessiner un rectangle sur l'image des symboles détectés

                matrice_symbole_x.append(int(x)) #ajout de la coordonnée x du symbole dans la matrice des coordonnées x des symboles
                matrice_symbole_y.append(int(y)) #ajout de la coordonnée y du symbole dans la matrice des coordonnées y des symboles
                matrice_symbole_type.append(1) #ajout du type du symbole dans la matrice des types des symboles
                n=n+1 

    #ENLEVER LES SYMBOLES FAUX POSITIFS ************************************************************************************************************************************************************************************
    
    points_et_type = np.array(np.column_stack((matrice_symbole_x, matrice_symbole_y, matrice_symbole_type)), dtype = np.int32)
    points = np.array(np.column_stack((matrice_symbole_x, matrice_symbole_y, )), dtype = np.int32)

    treshold = long_trait_max * 20 #distance pour que les points soient considérés comme des vraies symboles : cercle de rayon de 20 fois la longueur moyenne des traits centré sur le centre de gravité de la mire

    if points.size != 0: #si il y a des symboles détectés
        mean = np.mean(points, axis=0) #calcul de la moyenne des coordonnées des symboles
        distance = np.linalg.norm(points - mean, axis=1) #calcul de la distance entre chaque point et la moyenne
        good_points_et_type = points_et_type[distance < treshold] #on garde les points qui sont à moins de la distance treshold de la moyenne
        good_points = points[distance < treshold] 

        if len(good_points_et_type) >=15: #si il y a au moins 10 symboles proches, on considère qu'on a trouvé la mire
            hull = cv2.convexHull(np.array(good_points))
            cv2.polylines(frame, [hull], True, (0,255,0), 2) # dessiner le polygone convexe correspondant à la mire
            cv2.polylines(frame_symb_only, [hull], True, (0,255,0), 2) # dessiner le polygone convexe correspondant à la mire
            
            #dessiner le rectangle minimum qui englobe les points de contour
            rect = cv2.minAreaRect(good_points) 
            width_rectangle_mire, height_rectangle_mire = rect[1]
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(frame, [box], 0, (0,0,255), 2) 

        #TRI DES SYMBOLES & POSITION DE LA MIRE **************************************************************************************************************************************************************
            moments = cv2.moments(box)
            if rect is not None and moments["m00"] != 0 :
                #calcul des moments de l'image pour trouver son centre de gravité
                mean_x = int(moments["m10"] / moments["m00"]) 
                mean_y = int(moments["m01"] / moments["m00"])   
                cv2.circle(frame, (mean_x, mean_y), 10, (255, 255, 0), 2)
                cv2.putText(frame, "{}{}{}{}".format(" Position du centre de gravite de la mire : X=", mean_x," ; Y=", mean_y), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                angle = int(rect[-1]) #angle du rectangle englobant
                arrow_length = 100
                end_x = int(mean_x + arrow_length * np.cos(np.deg2rad(angle)))
                end_y = int(mean_y + arrow_length * np.sin(np.deg2rad(angle)))
                cv2.putText(frame, "{}{}".format(" Vecteur de rotation du rectangle englobant : angle = ", angle), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)         
                cv2.arrowedLine(frame, (mean_x, mean_y), (end_x, end_y), (255, 255, 0), 1)


        #MASQUE AVEC QUE LA MIRE ET CALCUL DE FLOU*******************************************************************************************************************************************************************************
    mode_name = ""
    if cv2.waitKey(1) == ord('r'): #on change de mode
        if mode ==0 : 
            mode = 1
        else:
            mode = 0
    if mode ==1: #
        mode_name = "Mise a jour des contours"
        box_2 = box
    else:
        mode_name = "Freeze des contours"


    cv2.fillConvexPoly(mire, box_2,255) #on remplit la mire avec du blanc
    mire = cv2.bitwise_and(frame_orig, frame_orig, mask=mire) #on applique la mire sur l'image de départ

    gray_mire = cv2.cvtColor(mire, cv2.COLOR_BGR2GRAY)
    fm_sobel = int(variance_of_image_blur_sobel(gray_mire))
    fm_laplacian = int(variance_of_image_blur_laplacian(gray_mire))
    fm_canny = int(variance_of_image_blur_Canny(gray_mire))

    cv2.putText(mire, "{}{}".format(" Sobel focus value : ", fm_sobel), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(mire, "{}{}".format(" Laplacian focus value : ", fm_laplacian), (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(mire, "{}{}".format(" Canny focus value : ", fm_laplacian), (0, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(mire, "{}{}".format(" Moyenne des valeurs de flou : ", int((fm_sobel+fm_laplacian+fm_canny)/3)), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(mire, "{}".format(mode_name), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
   
        #CREATION DE LA MATRICE SYMBOLE CORRESPONDANT A LA MIRE ET AUX BONS SYMBOLES*******************************************************************************************************************************

    if len(good_points_et_type) >=15: #si il y a des symboles
        if box is not None: #si la mire est détectée

            rotated_points = matrice_rotation(angle, mean_x, mean_y, good_points) #rotation de la matrice contenant les coordonnées des symboles
            normalised_points = matrice_normalisation(rotated_points, X_value_wanted, Y_value_wanted) #normalisation de la matrice contenant les coordonnées des symboles

        i=0
        for x, y in normalised_points:
            matrice_symb[round(x)][round(y)] = good_points_et_type[i, 2] #on qmet les symboles dans la nouvelle matrice normalisée et rotationnée
            i+=1 
        rotation_probable = comparison_mire(mire_reel, matrice_symb)
        cv2.putText(frame, "{}{}".format(" Angle de rotation probable de la mire : angle = ", rotation_probable + angle), (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)    
    #LIGNE CODE FIN************************************************************************************************************************************************************************************ 
 

    matrice_found_rgb_show(matrice_symb, X_value_wanted, Y_value_wanted) # transforme la matrice symbole en image RGB et l'affiche
    cv2.imshow('frame', frame) #on affiche l'image de base
    cv2.imshow('frame_symb_only', frame_symb_only) #on affiche l'image avec les symboles
    cv2.imshow('mire', mire) #on affiche la mire

    if cv2.waitKey(1) == ord('q'): #waitkey(0) --> only one frame, waitkey(1) --> video 
        break #si q est appuyé, on quitte la boucle

cap.release()
cv2.destroyAllWindows()


