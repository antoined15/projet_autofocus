#Camera Autofocus project : FIP-EII-2A Telecom Physique Strasbourg 2022-2023

#Authors :
#   - DOTTE Antoine
#   - JENNY Thibaud
#   - FISCHER Arnaud

#Date : 01/02/2023

#Description : This code is used to detect the symbols of the camera autofocus test chart and to display them in a matrix

import numpy as np
import cv2
import Reco_mire_fct as fct
import polig_4_cotes_intersect as polig_4C
import threading

appareil_utilise = "webcam" #webcam ou camera

if appareil_utilise == "camera":
    from picamera2 import Picamera2

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format":'RGB888',"size":(640,480)}))
    picam2.start()

else :
    cap = cv2.VideoCapture(0)

#####Variables globales##############################################################################################################
mat_dim_mire = np.zeros((15,15)) #Dimensions de la matrice
matrice_symb_moyenne =  np.zeros(np.shape(mat_dim_mire))

nbr_row_matrice = mat_dim_mire.shape[1] 
nbr_col_matrice = mat_dim_mire.shape[0]

num_pos_matrice_RGB_moy = 0 #variable qui permet de savoir quelle matrice de la liste matrice_RGB_moy on doit remplir
matrice_circulaire_mire_symbole = [] #liste qui contiendra les matrices symboles pour moyennage des symboles
nbr_moyennage_matrice_symbole = 30 #nombre de moyennage de la matrice des symboles : 30 c'est bien

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

def moyennage_matrice_symbole(matrice): #fonction qui permet de faire un moyennage des symboles détectés sur plusieurs images
    
    global num_pos_matrice_RGB_moy
    global matrice_circulaire_mire_symbole
    nbr_row_matrice = np.size(matrice, 0)
    nbr_col_matrice = np.size(matrice, 1)
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
            #print("moy_pixel_sans_0 = ", moy_pixel_sans_0)
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

###############################s####################################################################################################################################################################################################################

while True: 

    nbr_circle_detected = 0 #variable d'incrémetation
    long = 0 #longueur du symbole, sera utilisé pour la normalisation de la mire
    matrice_symb = np.zeros(np.shape(mat_dim_mire)) #on crée une matrice de 15x15 --> taille de la mire qui contiendra les symboles détectés
    matrice_symbole_x = [] #contient les coordonnées x des symboles de la mire trouvés
    matrice_symbole_y = [] #contient les coordonnées y des symboles de la mire trouvés
    matrice_symbole_type = [] #contient le type de symbole (1 = trait, 2 = cercle) trouvés
    matrice_symbole_rayon = [] #contient le rayon des cercles trouvés pour pouvoir faire la distinction des disques ou des ronds pleins

    mean_x = 0 #moyenne des coordonnées x des symboles de la mire trouvés
    mean_y = 0 #moyenne des coordonnées y des symboles de la mire trouvés
    angle = 0 #angle de rotation de la mire
    #LIGNE CODE DEBUT************************************************************************************************************************************************************************************
    
    #prendre une image
    if appareil_utilise == "camera":
        frame = picam2.capture_array()
    elif appareil_utilise == "webcam":
        ret, frame = cap.read()

    if frame is None:
        break

    #creation des images de travail
    frame_orig  = frame.copy() #copier l'image pour pouvoir la réutiliser pour la mire avec le masque
    frame_symb_only = cv2.cvtColor(np.full(frame.shape[:2], 0, dtype=np.uint8), cv2.COLOR_GRAY2RGB) #création d'une image noire sur laquelle sera intercallée les symboles détectés
    mire = np.full(frame.shape[:2], 0, dtype=np.uint8) #création d'une image noire sauf à l'endroit de la mire pour le calcul de la variance de l'image

    #détection des contours

    contours = fct.detect_contours(frame, contour_show = True) #détection des contours avec canny
    
    #DETECTION DES CERCLES ************************************************************************************************************************************************************************************
    for cnt in contours:    
        if cv2.contourArea(cnt) > 4: # Vérifier si le contour est suffisamment grand pour être considéré comme une forme

            rect = cv2.minAreaRect(cnt) # Obtenir le rectangle minimum qui englobe les points de contour
            (x, y), (w, h), angle = rect
            ratio = h/w
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
            if max(w, h) < long_trait_max*1.6 and (0.2 < ratio <0.5 or 1.8 < ratio <5):
                box = np.int32(cv2.boxPoints(rect))
                cv2.drawContours(frame_symb_only,[box],0,(0,0,255),2) #dessiner un rectangle sur l'image des symboles détectés

                matrice_symbole_x.append(int(x)) #ajout de la coordonnée x du symbole dans la matrice des coordonnées x des symboles
                matrice_symbole_y.append(int(y)) #ajout de la coordonnée y du symbole dans la matrice des coordonnées y des symboles
                matrice_symbole_type.append(1) #ajout du type du symbole dans la matrice des types des symboles
                matrice_symbole_rayon.append(0)
                nbr_circle_detected=nbr_circle_detected+1 

    #ENLEVER LES SYMBOLES FAUX POSITIFS ************************************************************************************************************************************************************************************
    
    points_et_type_et_rayon = np.array(np.column_stack((matrice_symbole_x, matrice_symbole_y, matrice_symbole_type, matrice_symbole_rayon)), dtype = np.int32)
    points = np.array(np.column_stack((matrice_symbole_x, matrice_symbole_y, )), dtype = np.int32)
  
    if points.size != 0: #si il y a des symboles détectés

        treshold = long_trait_max * 23 # distance jusqu'à laquelle les points sont considérés comme valide : si il sont à max 23*longueur_trait_max, on considère qu'ils font partie de la mire
        good_points, good_points_et_type_et_rayon = fct.suppr_symboles_detectes_non_pertinents(points, points_et_type_et_rayon, treshold)

        if len(good_points_et_type_et_rayon) >=10: # Si au moins 10 symboles sont détectés, on considère que la mire est détectée
            hull = cv2.convexHull(np.array(good_points)) #dessiner le contour convexe qui englobe les points de contour 
            rect = cv2.minAreaRect(good_points) #dessiner le rectangle minimum qui englobe les points de contour
            box = np.int32(cv2.boxPoints(rect)) #obtenir les 4 coins du rectangle englobant

            #print("hull --> ", hull)

            #utilisation de l'algo pour détecter les 4 cotés du rectangle --> pas très efficace
            #flat_hull = []
            #for i in range(len(hull)):
                #x = hull[i][0][0]
                #y = hull[i][0][1]
                #flat_hull.append([x, y])
            #print("flat_hull --> ", flat_hull)
            #polig_4C.algo_4cotes(flat_hull, img_show = False)
            #polyg4c_approx = polig_4C.approxpoly(polig_4C.polyg)
            #print("longueur polyg4c_approx", len(polyg4c_approx), "\t polyg4c_approx --> ", polyg4c_approx)
            #cv2.polylines(frame, [np.array(polyg4c_approx)], True, (100,255,255), 2) # dessiner le polygone convexe correspondant à la mire

            #POSITION DE LA MIRE *************************************************************************************************************************************************************
            moments = cv2.moments(box)
            
            #calcul des moments de l'image pour trouver son centre de gravité
            mean_x = int(moments["m10"] / moments["m00"]) 
            mean_y = int(moments["m01"] / moments["m00"])   
                
            angle = int(rect[-1]) #angle du rectangle englobant
            if angle == 90: angle = 0
            arrow_length = 100
            end_x = int(mean_x + arrow_length * np.cos(np.deg2rad(angle)))
            end_y = int(mean_y + arrow_length * np.sin(np.deg2rad(angle)))   
            cv2.circle(frame_symb_only, (mean_x, mean_y), int(treshold), (0,100, 0), 2) #dessiner un cercle sur l'image des symboles détectés 

            #CREATION DE LA MATRICE SYMBOLE CORRESPONDANT A LA MIRE ET AUX BONS SYMBOLES*******************************************************************************************************************************
 
            rotated_points = fct.matrice_rotation(angle, mean_x, mean_y, good_points) #rotation de la matrice contenant les coordonnées des symboles
            normalised_points = fct.matrice_normalisation(rotated_points, nbr_col_matrice, nbr_col_matrice ) #normalisation de la matrice contenant les coordonnées des symboles

            i=0
            matrice_sequence_détectée =  [[[0 for i in range(3)] for j in range(15)] for k in range(15)] #mise en forme des données pour les séquences
            for x, y in normalised_points: #on met les symboles dans la nouvelle matrice normalisée et rotationnée
                x_mat = int(np.around(x))
                y_mat = int(np.around(y))

                if matrice_symb[x_mat][y_mat] == 0 : #Si la case est vide
                    matrice_symb[x_mat][y_mat] = good_points_et_type_et_rayon[i, 2] #on met le symbole
                    matrice_sequence_détectée[x_mat][y_mat][0] = matrice_symb_moyenne[x_mat][y_mat]
                    matrice_sequence_détectée[x_mat][y_mat][1] = good_points_et_type_et_rayon[i, 0]
                    matrice_sequence_détectée[x_mat][y_mat][2] = good_points_et_type_et_rayon[i, 1]
                    
                elif matrice_symb[x_mat][y_mat] == 2 : #si la case est déjà un rond
                    matrice_symb[x_mat][y_mat] = 3 #on met un cercle
                    matrice_sequence_détectée[x_mat][y_mat][0] = matrice_symb_moyenne[x_mat][y_mat]
                    matrice_sequence_détectée[x_mat][y_mat][1] = good_points_et_type_et_rayon[i, 0]
                    matrice_sequence_détectée[x_mat][y_mat][2] = good_points_et_type_et_rayon[i, 1]
                i = i+1


    
    #DETECTION DES APPARIEMENTS ENTRE LES SYMBOLES#####################################################################################################

    matrice_symb_moyenne = moyennage_matrice_symbole(matrice_symb) #retourne la matrice symbole moyennée sur 5 images --> permet de mieux connaitre la matrice symbole

    try : matrice_sequence_détectée
    except NameError: matrice_sequence_détectée = None
    if matrice_sequence_détectée is not None: #si il y a au moins 10 symboles, on détecte l'angle de rotation de la mire et on cherche les appariements
        #rotation_probable = fct.comparison_mire(Mire_reel, matrice_symb_moyenne) #pas utilisé car les apariements marchent mieux
        nbr_erreur_seq_max = 1
        appariements, rotation_probable, erreur_moy_appariement = fct.appariement_symboles_4rotations(matrice_sequence_détectée, Mire_reel, nbr_erreur_seq_max)
        nbr_appariements = len(appariements)
        #for point in appariements:
            #print("position réelle", point[0], "\tposition détectée", point[1], "\tséquence", point[2], "\t symbole correspondant", Mire_reel[point[0][0]][point[0][1]], "\t nbr erreur", point[3])
        #print("rotation probable", rotation_probable, "\t erreur moyenne appariement", erreur_moy_appariement, "\t nombre d'appariements", nbr_appariements)

        

        
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

    #AFFICHAGE SUR L'IMAGE "FRAME" ************************************************************************************************************************************************************************************
    try: good_points_et_type_et_rayon
    except NameError: good_points_et_type_et_rayon = None
   

    if  good_points_et_type_et_rayon is not None and len(good_points_et_type_et_rayon) >=10 : #si il y a au moins 10 symboles détectés
        cv2.putText(frame, "{}{}".format(" Angle de rotation probable de la mire : angle = ", rotation_probable + int(angle)), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA) 
        cv2.circle(frame, (mean_x, mean_y), 10, (255, 255, 0), 2)
        cv2.putText(frame, "{}{}{}{}".format(" Position du centre de gravite de la mire : X=", mean_x," ; Y=", mean_y), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  
        cv2.putText(frame, "{}{}".format(" Vecteur de rotation du rectangle englobant : angle = ", int(angle)), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)         
        cv2.arrowedLine(frame, (mean_x, mean_y), (end_x, end_y), (255, 255, 0), 1)
        cv2.putText(frame, "{}{}".format(" Nombre d'appariements trouves : ", nbr_appariements), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  
        cv2.putText(frame, "{}{}".format(" Erreur moyenne par d'appariement [symboles/squence] : ", round(erreur_moy_appariement, 2)), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)       

        angle_total = rotation_probable + angle #angle du rectangle englobant
        arrow_length = 150
        end_x_angle_tot = int(mean_x + arrow_length * np.cos(np.deg2rad(angle_total)))
        end_y_angle_tot = int(mean_y + arrow_length * np.sin(np.deg2rad(angle_total)))  

        cv2.arrowedLine(frame, (mean_x, mean_y), (end_x_angle_tot, end_y_angle_tot), (50, 90, 255), 1)
        cv2.polylines(frame, [hull], True, (0,255,0), 2) # dessiner le polygone convexe correspondant à la mire
        cv2.polylines(frame_symb_only, [hull], True, (0,255,0), 2) # dessiner le polygone convexe correspondant à la mire
        cv2.drawContours(frame, [box], 0, (0,0,255), 2) 


    #AFFICHAGE SUR L'IMAGE "MIRE" ************************************************************************************************************************************************************************************
    if  (good_points_et_type_et_rayon is not None and len(good_points_et_type_et_rayon) >=10) or mode == 0 : #si il y a au moins 10 symboles détectésor mode == 0 : #si il y a au moins 15 symboles
        cv2.fillConvexPoly(mire, box_if_freeze,255) #on remplit la mire avec du blanc
        mire = cv2.bitwise_and(frame_orig, frame_orig, mask=mire) #on applique la mire sur l'image de départ

        gray_mire = cv2.cvtColor(mire, cv2.COLOR_BGR2GRAY)
        fm_sobel = int(fct.variance_of_image_blur_sobel(gray_mire))
        fm_laplacian = int(fct.variance_of_image_blur_laplacian(gray_mire))
        fm_canny = int(fct.variance_of_image_blur_Canny(gray_mire))
        fm_entropie = round(fct.variance_of_image_blur_entropy(gray_mire), 4)
        fm_corner_counter = fct.variance_of_image_blur_corner_counter(gray_mire)
        fm_picture_variance = round(fct.variance_of_image_blur_picture_variability(gray_mire), 3)

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

    try : appariements
    except NameError : appariements = None


    fct.matrice_rgb_show(matrice_symb_moyenne, appariements, affich_err_appariement = True ) # transforme la matrice symbole en image RGB et l'affiche
    cv2.imshow('frame', frame) #on affiche l'image de base
    cv2.imshow('frame_symb_only', frame_symb_only) #on affiche l'image avec les symboles
    cv2.imshow('mire', mire) #on affiche la mire

    if cv2.waitKey(1) == ord('q'): #waitkey(0) --> only one frame, waitkey(1) --> video 
        break #si q est appuyé, on quitte la boucle

cv2.destroyAllWindows()

if appareil_utilise == "webcam":
    cap.release()
elif appareil_utilise == "camera":
    picam2.stop()

