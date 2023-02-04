
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

cap = cv2.VideoCapture(0)

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



def variance_of_image_blur_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	return round(cv2.Laplacian(image, cv2.CV_64F).var(), 2)

def variance_of_image_blur_sobel(image):
	# compute the Sobel of the image and then return the focus
	return round(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3 ).var(), 1)

def variance_of_image_blur_Canny(image):
	# compute the Canny of the image and then return the focus
	return round(cv2.Canny(image, 100, 200).var(), 1)

def symbole_identique_2_matrices(matrice1, matrice2): #calcule le nombre de symboles identiques entre deux matrices pour la même position

    symbole_validated = 0
    for i in range(15):
        for j in range(15):
            if matrice1[i][j]==matrice2[i][j]:
                symbole_validated = symbole_validated + 1

    return round(100 * symbole_validated/225, 3)


def comparison_mire(real_mire, matrice_symb): #fonction globale pour comparer la mire originelle avec la mire potentielle trouvée

    symbole_found = np.count_nonzero(np.array(matrice_symb))
    pourcent_symb_detect = round(100 * symbole_found/225, 3)
    print("Pourcentage de symboles détectés : ", pourcent_symb_detect, "%")

    matrice_symb_rot_90 = np.rot90(matrice_symb, k=1)
    matrice_symb_rot_180 = np.rot90(matrice_symb, k=2)
    matrice_symb_rot_270 = np.rot90(matrice_symb, k=3)

    pourcent_symb_valide_0 = symbole_identique_2_matrices(real_mire, matrice_symb)
    pourcent_symb_valide_90 = symbole_identique_2_matrices(real_mire, matrice_symb_rot_90)
    pourcent_symb_valide_180 = symbole_identique_2_matrices(real_mire, matrice_symb_rot_180)
    pourcent_symb_valide_270 = symbole_identique_2_matrices(real_mire, matrice_symb_rot_270)

    print(matrice_symb)
    print("Pourcentage de symboles validés : r=0° : ",pourcent_symb_valide_0, "%")
    print("Pourcentage de symboles validés : r=90° : ",pourcent_symb_valide_90, "%")
    print("Pourcentage de symboles validés : r=180° : ",pourcent_symb_valide_180, "%")
    print("Pourcentage de symboles validés : r=270° : ",pourcent_symb_valide_270, "%")

    max_symb_valid = max(pourcent_symb_valide_0, pourcent_symb_valide_90, pourcent_symb_valide_180, pourcent_symb_valide_270)
    if max_symb_valid == pourcent_symb_valide_0:
        print("Rotation de la mire probable : 0°")
        rot_probable = 0
    elif max_symb_valid == pourcent_symb_valide_90:
        print("Rotation de la mire probable : 90°")
        rot_probable = 90
    elif max_symb_valid == pourcent_symb_valide_180:
        print("Rotation de la mire probable : 180°")
        rot_probable = 180
    else:
        print("Rotation de la mire probable : 270°")
        rot_probable = 270


    print("----------------")
    return rot_probable
 #SI LIGNES OU COLONNES VIDES DANS LA MATRICE TROUVEE,ALORS CA VEUT DIRE QUE LA MIRE N'EST PAS COMPLETE--> DECALER LES MOTIFS EN CONSEQUENCE --> REFAIRE LA NORMALISATION EN FCT DE LA LONG ET LARGEUR DU RECTANGLE ENGLOBANT
 #SI MIRE =0, ALORS ON SAIS APS
 #REMPLACER LES RONDS OU CERCLES PAR DES VIDES
 #ESSAYER DE TROUVER LA MIRE MAIS SEULEMENT AVEC LES RECTANGLES ? OU EN FAISANT DES MATRICES 4*4 ET VOIR SI C'EST IDENTIQUE
#faire une image avec opencv pour voir ce que la matrice renvoie avec des ronds de couleur par exemple


while True: 
    #LIGNE CODE DEBUT************************************************************************************************************************************************************************************
    ret, frame = cap.read() #prendre une image

    #creation des images de travail
    frame_orig  = frame.copy() #copier l'image pour pouvoir la réutiliser pour la mire avec le masque
    frame_symb_only = cv2.cvtColor(np.full(frame.shape[:2], 0, dtype=np.uint8), cv2.COLOR_GRAY2RGB) #création d'une image noire sur laquelle sera intercallée les symboles détectés
    mire = np.full(frame.shape[:2], 0, dtype=np.uint8) #création d'une image noire sauf à l'endroit de la mire pour le calcul de la variance de l'image

    #détection des contours
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convertir en noir et blanc
    canny_image = cv2.Canny(frame, 50, 150) #Appliquer un filtre de Canny pour détecter les contours
    contours, _ = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #trouver les contours de l'image avec le filtre de Canny
    canny_image_cont = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB) #convertir en couleur pour pouvoir afficher les contours en couleur --> frame avec contours
    cv2.imshow('contours', canny_image_cont) #on affiche l'image avec les contours
   
   
    n = 0 #variable d'incrémetation
    long = 0 #longueur du symbole, sera utilisé pour la normalisation de la mire
    matrice_symb = np.zeros((15, 15), dtype=int) #on crée une matrice de 15x15 --> taille de la mire qui contiendra les symboles détectés
    matrice_symbole_x = [] #contient les coordonnées x des symboles de la mire trouvés
    matrice_symbole_y = [] #contient les coordonnées y des symboles de la mire trouvés
    matrice_symbole_type = [] #contient le type de symbole (1 = trait, 2 = cercle) trouvés

    #DETECTION DES CERCLES ************************************************************************************************************************************************************************************
    for cnt in contours:    
        if cv2.contourArea(cnt) > 5: # Vérifier si le contour est suffisamment grand pour être considéré comme une forme

            rect = cv2.minAreaRect(cnt) # Obtenir le rectangle minimum qui englobe les points de contour
            (x, y), (w, h), angle = rect
            ratio = h/w
            if ratio == 1:  #détection des cercles : on considère que c'est des carrés parfaits donc ratio = 1 (plus facile à détecter que des cercles eux mêmes)
                box = np.int0(cv2.boxPoints(rect))
                
                cv2.circle(frame_symb_only, (int(x), int(y)), int(min(w/2,h/2)), (255, 0, 0), 2) #dessiner un cercle sur l'image des symboles détectés

                long = max(w, h) + long #longueur du symbole
                matrice_symbole_x.append(int(x)) #ajout de la coordonnée x du symbole dans la matrice des coordonnées x des symboles
                matrice_symbole_y.append(int(y)) #ajout de la coordonnée y du symbole dans la matrice des coordonnées y des symboles
                matrice_symbole_type.append(2) #ajout du type du symbole dans la matrice des types des symboles
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
            if max(w, h) < long_trait_max*1.5 and (0.15 < ratio <0.40 or 2 < ratio <10): #les rectangles sont environ 5fois plus grand en longueur que el largeur : ratio = env 3.5 ou env 0.28
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

        if len(good_points_et_type) >=10: #si il y a au moins 10 symboles proches, on considère qu'on a trouvé la mire
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
            
            if box is not None: #si la mire est détectée
                cv2.fillConvexPoly(mire, box,255) #on remplit la mire avec du blanc
                mire = cv2.bitwise_and(frame_orig, frame_orig, mask=mire) #on applique la mire sur l'image de départ

                gray_mire = cv2.cvtColor(mire, cv2.COLOR_BGR2GRAY)
                fm_sobel = int(variance_of_image_blur_sobel(gray_mire))
                fm_laplacian = int(variance_of_image_blur_laplacian(gray_mire))
                fm_canny = int(variance_of_image_blur_Canny(gray_mire))
            else:
                fm_sobel = 0
                fm_laplacian = 0
                fm_canny = 0

            cv2.putText(mire, "{}{}".format(" Sobel focus value : ", fm_sobel), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(mire, "{}{}".format(" Laplacian focus value : ", fm_laplacian), (0, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(mire, "{}{}".format(" Canny focus value : ", fm_laplacian), (0, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(mire, "{}{}".format(" Moyenne des valeurs de flou : ", int((fm_sobel+fm_laplacian+fm_canny)/3)), (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
   
   
        #CREATION DE LA MATRICE SYMBOLE CORRESPONDANT A LA MIRE ET AUX BONS SYMBOLES*******************************************************************************************************************************

            #matrice de rotation des symboles 
            if box is not None: #si la mire est détectée
                angle_rad = math.radians(-angle) #on convertit l'angle en radians

                rotation_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]]) #on crée la matrice de rotation
                centre_matrix = np.array([mean_x, mean_y])
                rotated_points = []
                for point in good_points:
                    points = point - centre_matrix
                    rotated_point = np.dot(rotation_matrix, points)
                    rotated_points.append(rotated_point)

            #normalisation de la matrice de symbole #MARCHE car on obtient une matrice de 15x15
                rotated_points_x = [point[0] for point in rotated_points]
                rotated_points_y = [point[1] for point in rotated_points]

                min_x = np.min(rotated_points_x)
                min_y = np.min(rotated_points_y)
                max_x = np.max(rotated_points_x)
                max_y = np.max(rotated_points_y)

                rotated_points_x = ((rotated_points_x - min_x)/(max_x - min_x)) *14 #on normalise les coordonnées des points pour qu'ils soient compris entre 0 et 15
                rotated_points_y = ((rotated_points_y - min_y)/(max_y - min_y)) * 14 ##on normalise les coordonnées des points pour qu'ils soient compris entre 0 et 15


                rotated_points = np.transpose(np.array([rotated_points_x, rotated_points_y]))
                #pour chaque point correspondant à des symboles --> On rotationne les symboles en ftc de leur position, de la distance avec le centre et de la matrice de rotation
                i=0
                for x, y in rotated_points:
                    matrice_symb[round(x)][round(y)] = good_points_et_type[i, 2] #on met un 1 dans la matrice de symbole
                    i+=1 

                rotation_probable = comparison_mire(mire_reel, matrice_symb)
                cv2.putText(frame, "{}{}".format(" Angle de rotation probable de la mire : angle = ", rotation_probable + angle), (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)    
    #LIGNE CODE FIN************************************************************************************************************************************************************************************ 
    #Affiche l'image'
        colors = [(0, 0, 0), (0, 0, 255), (0, 255, 0)]
        matrice_symb_rgb = np.zeros((15, 15, 3), dtype=np.uint8)
        for i in range(15):
            for j in range(15):
                try : 
                    matrice_symb_rgb[j, i] = colors[matrice_symb[i, j]]
                except:
                    matrice_symb_rgb[i, j] = (0, 0, 0)
        matrice_symb_rgb = cv2.resize(matrice_symb_rgb, (400, 400), cv2.INTER_LINEAR)
    
    cv2.imshow('frame', frame) #on affiche l'image de base
    cv2.imshow('frame_symb_only', frame_symb_only) #on affiche l'image avec les symboles
    cv2.imshow('mire', mire) #on affiche la mire
    cv2.imshow('symboles detectes', matrice_symb_rgb) #on affiche la matrice de symbole
    if cv2.waitKey(1) == ord('q'): #waitkey(0) --> only one frame, waitkey(1) --> video 
        break #si q est appuyé, on quitte la boucle

cap.release()
cv2.destroyAllWindows()


            #print(matrice_symb.shape)
            #while np.all(matrice_symb[:,-1] == 0): #si la dernière colonne est vide
                #matrice_symb = np.delete(matrice_symb, -1, 1) #on supprime la dernière colonne
            #while np.all(matrice_symb[-1,:] == 0):
                #matrice_symb = np.delete(matrice_symb, -1, 0) #on supprime la dernière ligne
            #print(matrice_symb.shape)         
            #print(matrice_symb)