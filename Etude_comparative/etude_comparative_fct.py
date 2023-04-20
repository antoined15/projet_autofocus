import time
import numpy as np
import random
import cv2
from Focuser import Focuser
from picamera2 import Picamera2
from pantilt import Pantilt


def position_tourelle(pt,X, Y):
	xmap = pt.map(X,0,180,200,800)
	ymap = pt.map(Y,0,90,310,650)
	for i in range(5):
		pt.position(xmap,ymap)
		time.sleep(0.1)
	time.sleep(2)
	print("Tourelle déplacée : \tmoteur 1 = ", X, " ; \tmoteur 2 = ", Y)

def position_moteur_flou(focuser, type_camera,  pos_M):
	tempo = 0.2 #temps d'attente entre chaque modification de la position du moteur autofocus
	if type_camera == "64mp" or "16mp":
		#Programme de modification de la position du moteur autofocus pour les arducam --> A TESTER
		focuser.set(Focuser.OPT_FOCUS,pos_M)
		time.sleep(tempo)
	else:
		pass

	#attente que le déplacement soit terminé
	#print("Moteur autofocus déplacé : \tposition = ", pos_M)
	pass

def trie_liste(x,y1,y2,y3,y4,y5,y6 = []):

	x_y1 = list(zip(x,y1))
	x_y1 = sorted(x_y1, key=lambda l: l[0])
	y1 = [t[1] for t in x_y1]

	x_y2 = list(zip(x,y2))
	x_y2 = sorted(x_y2, key=lambda l: l[0])
	y2 = [t[1] for t in x_y2]

	x_y3 = list(zip(x,y3))
	x_y3 = sorted(x_y3, key=lambda l: l[0])
	y3 = [t[1] for t in x_y3]

	x_y4 = list(zip(x,y4))
	x_y4 = sorted(x_y4, key=lambda l: l[0])
	y4 = [t[1] for t in x_y4]

	x_y5 = list(zip(x,y5))
	x_y5 = sorted(x_y5, key=lambda l: l[0])
	y5 = [t[1] for t in x_y5]

	if y6 != []:
		x_y6 = list(zip(x,y6))
		x_y6 = sorted(x_y6, key=lambda l: l[0])
		y6 = [t[1] for t in x_y6]

	x = [t[0] for t in x_y1]

	if y6 != []:
		return x ,y1 ,y2, y3, y4, y5, y6
	else:
		return x ,y1 ,y2, y3, y4, y5
	

def detection_cercles(contours, canny_image_cont):

	matrice_cercle = [[], [], []] #matrice des coordonnées des cercles détectés [[x1, x2, x3, ...], [y1, y2, y3, ...]]
	n= 0 #nombre de cercles détectés
	long = 0 #longueur moyenne des symboles détectés

	for cnt in contours:    
		if cv2.contourArea(cnt) > 5: # Vérifier si le contour est suffisamment grand pour être considéré comme une forme

			rect = cv2.minAreaRect(cnt) # Obtenir le rectangle minimum qui englobe les points de contour
			(x, y), (w, h), angle = rect
			ratio = h/w
			if 0.999 < ratio <1.001 :  #détection des cercles : on considère que c'est des carrés parfaits donc ratio = 1 (plus facile à détecter que des cercles eux mêmes)
				long = max(w, h) + long #longueur du symbole
				matrice_cercle[0].append(int(x)) #ajout de la coordonnée x du symbole dans la matrice des coordonnées x des symboles
				matrice_cercle[1].append(int(y)) #ajout de la coordonnée y du symbole dans la matrice des coordonnées y des symboles 
				matrice_cercle[2].append(2) #ajout de la valeur 1 dans la matrice des valeurs des symboles (2 pour un cercle)
				cv2.circle(canny_image_cont, (int(x), int(y)), int(min(w/2,h/2)), (255,0,0), 2) #dessiner un cercle sur l'image des symboles détectés          
				n = n+1
	if n>0:
		long_trait_max = long/n #calcul de la longueur moyenne des traits de la mire
	else:
		long_trait_max = 0

	return n, matrice_cercle, long_trait_max	

def detection_traits(contours, canny_image_cont, long_trait_max):
	matrice_trait = [[], [], []] #matrice des coordonnées des traits détectés [[x1, x2, x3, ...], [y1, y2, y3, ...]]
	n=0 #nombre de traits détectés

	for cnt in contours:     
        # Vérifier si le contour est suffisamment grand pour être considéré comme une forme
		if cv2.contourArea(cnt) > 5: # Obtenir le rectangle minimum qui englobe les points de contour
			rect = cv2.minAreaRect(cnt)
			(x, y), (w, h), angle = rect
			ratio = h/w
			if max(w, h) < long_trait_max*1.5 and (0.10 < ratio <0.5 or 2 < ratio <10): #les rectangles sont environ 5fois plus grand en longueur que el largeur : ratio = env 3.5 ou env 0.28
				box = np.int0(cv2.boxPoints(rect))
				cv2.drawContours(canny_image_cont,[box],0,(0,0,255),2) #dessiner un rectangle sur l'image des symboles détectés

				matrice_trait[0].append(int(x)) #ajout de la coordonnée x du symbole dans la matrice des coordonnées x des symboles
				matrice_trait[1].append(int(y))#ajout de la coordonnée y du symbole dans la matrice des coordonnées y des symboles
				matrice_trait[2].append(1) #ajout de la valeur 1 dans la matrice des valeurs des symboles (1 pour un trait)
				n=n+1 

	return n, matrice_trait,

def contours_image(frame):

	#DETERMINATION DES CONTOURS DE L'IMAGE
	canny_image = cv2.Canny(frame, 50,150) #Appliquer un filtre de Canny pour détecter les contours
	contours, _ = cv2.findContours(canny_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #trouver les contours de l'image avec le filtre de Canny
	canny_image_cont = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2RGB) #convertir en couleur pour pouvoir afficher les contours en couleur --> frame avec contours
	return contours, canny_image_cont



def retrait_faux_positifs(matrice_symb, long_symb):

	#matrice_symb = np.array(matrice_symb)
	matrice_coord = np.array(np.column_stack((matrice_symb[0], matrice_symb[1])), dtype = np.int32)#matrice des coordonnées des symboles [[x1, x2, x3, ...], [y1, y2, y3, ...]]
	treshold = long_symb * 23
	matrice_symb_good = [[], [], []]
	mean = np.mean(matrice_coord, axis=0) #calcul de la moyenne des coordonnées des symboles
	distance = np.linalg.norm(matrice_coord - mean, axis=1) 

	for point in range (len(distance)):
		if distance[point] < treshold:
			matrice_symb_good[0].append(matrice_symb[0][point])
			matrice_symb_good[1].append(matrice_symb[1][point])
			matrice_symb_good[2].append(matrice_symb[2][point])

	return 	matrice_symb_good 


def position_angle_mire(matrice_symb, image_contour):
	matrice_coord = np.array(np.column_stack((matrice_symb[0], matrice_symb[1])), dtype = np.int32)
	hull = cv2.convexHull(np.array(matrice_coord))
	cv2.polylines(image_contour, [hull], True, (0,255,0), 2) # dessiner le polygone convexe correspondant à la mire
	            
	#dessiner le rectangle minimum qui englobe les points de contour
	rect = cv2.minAreaRect(matrice_coord) 
	box = np.int0(cv2.boxPoints(rect))
	cv2.drawContours(image_contour, [box], 0, (0,0,255), 2) 
	moments = cv2.moments(box)

	#calcul des moments de l'image pour trouver son centre de gravité
	mean_x = int(moments["m10"] / moments["m00"]) 
	mean_y = int(moments["m01"] / moments["m00"])   
	cv2.circle(image_contour, (mean_x, mean_y), 10, (255, 255, 0), 2)
	cv2.putText(image_contour, "{}{}{}{}".format(" Position du centre de gravite de la mire : X=", mean_x," ; Y=", mean_y), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
	angle = int(rect[-1]) #angle du rectangle englobant
	arrow_length = 100
	end_x = int(mean_x + arrow_length * np.cos(np.deg2rad(angle)))
	end_y = int(mean_y + arrow_length * np.sin(np.deg2rad(angle)))
	cv2.putText(image_contour, "{}{}".format(" Vecteur de rotation du rectangle englobant : angle = ", angle), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)         
	cv2.arrowedLine(image_contour, (mean_x, mean_y), (end_x, end_y), (255, 255, 0), 1)

	return mean_x, mean_y, angle, box


def nbre_symboles_mire_detectes(frame) : 

	matrice_symb_MIRE = [[], []]
	box_mire = [[0, 0], [0, 0], [0, 0], [0, 0]] #coordonnées de la boite englobante de la mire 
	contours, canny_image_cont = contours_image(frame)
	nbr_cercles, matrice_symb_cercle, long_symb = detection_cercles(contours, canny_image_cont)
	nbr_traits, matrice_symb_trait = detection_traits(contours, canny_image_cont, long_symb)
	mean_X_mire = 0
	mean_Y_mire = 0
	angle = 0

	for i in range(len(matrice_symb_cercle[0])): #fusion des 2 listes de symboles
		matrice_symb_trait[0].append(matrice_symb_cercle[0][i])
		matrice_symb_trait[1].append(matrice_symb_cercle[1][i])
		matrice_symb_trait[2].append(matrice_symb_cercle[2][i])
	matrice_symb = matrice_symb_trait


	if len(matrice_symb[0]) >= 15: #si on détecte au moins 10 symboles
		matrice_symb_MIRE = retrait_faux_positifs(matrice_symb, long_symb) #on retire les faux positifs (symboles détectés mais qui ne sont pas dans la mire

		if len(matrice_symb_MIRE[0]) >=15: #si on a détecté au moins 15 symboles, on a trouvé la mire
			mean_X_mire, mean_Y_mire, angle, box_mire = position_angle_mire(matrice_symb_MIRE, canny_image_cont)
	cv2.imshow('Contours & mire', canny_image_cont) #on affiche l'image avec les contours, les symboles et le rectangle englobant la mire 

	return (len(matrice_symb_MIRE[0])), box_mire, mean_X_mire, mean_Y_mire, angle

def nbre_symboles_mires_detectes_moyenne(nbre_moy, cap, pos_T, pos_M, color_frame, taille_text_frame):
	

	nbr_symboles = []
	mean_X_mire = []
	mean_Y_mire = []
	angle_mire = []
	box_mire = []

	for i in range(nbre_moy): 
		frame = cap.capture_array() #prendre une image
		nbr_symb, box, mean_x, mean_y, angle = nbre_symboles_mire_detectes(frame) #on compte le nombre de symboles de la mire détectés 
		nbr_symboles.append(nbr_symb)
		mean_X_mire.append(mean_x)
		mean_Y_mire.append(mean_y)
		angle_mire.append(angle)
		box_mire.append(box)


		if cv2.waitKey(1) == ord('q'): 
			break #si q est appuyé, on quitte la boucle

		cv2.putText(frame, "{}{}{}{}".format(" Position de la tourelle :  ",pos_T[0],";", pos_T[1]), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)  
		cv2.putText(frame, "{}".format(" Recherche de la position de la mire"), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)  
		cv2.putText(frame, "{}{}{}{}".format(" Prise d'image : ", i+1, "/", nbre_moy), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)  
		cv2.putText(frame, "{}{}".format(" Position du moteur autofocus :  ",pos_M), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA) 
		cv2.imshow('frame', frame) #on affiche l'image de base


	nbr_symb_moy = int(np.mean(nbr_symboles))
	mean_X_mire_moy = int(np.mean(mean_X_mire))
	mean_Y_mire_moy = int(np.mean(mean_Y_mire))
	angle_mire_moy = int(np.mean(angle_mire))
	box_mire_moy = np.round(np.mean(box_mire, axis = 0)).astype(np.int32)

	return nbr_symb_moy, box_mire_moy, mean_X_mire_moy, mean_Y_mire_moy, angle_mire_moy





def img_masque_mire(frame, box_mire):

	mire = np.full(frame.shape[:2], 0, dtype=np.uint8) #création d'une image noire sauf à l'endroit de la mire pour le calcul de la variance de l'image
	cv2.fillConvexPoly(mire, box_mire,255) #on remplit la mire avec du blanc
	mire = cv2.bitwise_and(frame, frame, mask=mire) #on applique la mire sur l'image de départ

	cv2.imshow('mire', mire) #on affiche la mire masquée

	return mire



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

def find_center(box):
	x_sum, y_sum =0 ,0

	for point in box:
		x_sum += point[0]
		y_sum += point[1]
	
	return x_sum / len(box), y_sum / len(box)

def enlarge_square(box, scale_factor):
	cx, cy = find_center(box)
	enlarged_box = []

	for point in box:
		x = cx + (point[0] - cx)*scale_factor
		y = cy + (point[1] - cy)*scale_factor
		enlarged_box.append([x,y])
	
	return enlarged_box


def variance_of_image_blur(frame,best_box, color_frame, taille_text_frame, pos_T, pos_M):
	mire_masque = img_masque_mire(frame, best_box) 	#ON MET LE MASQUE SUR LA MIRE TROUVEE


	gray_mire = cv2.cvtColor(mire_masque, cv2.COLOR_BGR2GRAY)
	fm_sobel = int(variance_of_image_blur_sobel(gray_mire))
	fm_laplacian = int(variance_of_image_blur_laplacian(gray_mire))
	fm_canny = int(variance_of_image_blur_Canny(gray_mire))
	fm_entropy = int(variance_of_image_blur_entropy(gray_mire))
	fm_corner_counter = int(variance_of_image_blur_corner_counter(gray_mire))
	fm_picture_variability = int(variance_of_image_blur_picture_variability(gray_mire))


	return fm_sobel, fm_laplacian, fm_canny, fm_entropy, fm_corner_counter, fm_picture_variability


def variance_of_image_blur_moyenne(cap,best_box, nbr_mesures, color_frame, taille_text_frame, pos_T, pos_M):

	focus_sobel_tab = []
	focus_laplacian_tab = []
	focus_canny_tab = []
	focus_entropy_tab = []
	focus_corner_counter_tab = []
	focus_picture_variability_tab = []
	fm_sobel_moy = 0
	fm_laplacian_moy = 0
	fm_canny_moy = 0
	fm_entropy_moy = 0
	fm_corner_counter_moy = 0
	fm_picture_variability_moy = 0

	for i in range(nbr_mesures):
		frame = cap.capture_array() #prendre une image

		fm_sobel, fm_laplacian, fm_canny, fm_entropy, fm_corner_counter, fm_picture_variability = variance_of_image_blur(frame, best_box,color_frame, taille_text_frame, pos_T, pos_M)

		cv2.putText(frame, "{}{}{}{}".format(" Position de la tourelle :  ",pos_T[0],";", pos_T[1]), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)  
		cv2.putText(frame, "{}".format(" Position de la mire trouvee, calcul du flou"), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA) 
		cv2.putText(frame, "{}{}{}{}".format(" Prise d'image : ", i+1, "/", nbr_mesures), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)
		cv2.putText(frame, "{}{}".format(" Position du moteur autofocus :  ",pos_M), (0, 80), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)
		cv2.putText(frame, "{}{}".format(" Valeur de flou de la mire avec Sobel : ", fm_sobel), (0, 100), cv2.FONT_HERSHEY_SIMPLEX,taille_text_frame, color_frame, 1, cv2.LINE_AA)
		cv2.putText(frame, "{}{}".format(" Valeur de flou de la mire avec Laplace : ", fm_laplacian), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)
		cv2.putText(frame, "{}{}".format(" Valeur de flou de la mire avec Canny : ", fm_canny), (0, 140), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)
		cv2.putText(frame, "{}{}".format(" Entropie de l'image : ", fm_entropy), (0, 160), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)
		cv2.putText(frame, "{}{}".format(" Nombre de coins détectés: ", fm_corner_counter), (0, 180), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)
		cv2.putText(frame, "{}{}".format(" Variabilité de l'image : ", fm_picture_variability), (0, 200), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)


		cv2.imshow('frame', frame) #on affiche l'image de base

		focus_sobel_tab.append(fm_sobel)
		focus_laplacian_tab.append(fm_laplacian)
		focus_canny_tab.append(fm_canny)
		focus_entropy_tab.append(fm_entropy)
		focus_corner_counter_tab.append(fm_corner_counter)
		focus_picture_variability_tab.append(fm_picture_variability)
		if cv2.waitKey(1) == ord('q'):
			break #si q est appuyé, on quitte la boucle

	fm_sobel_moy = int(np.mean(focus_sobel_tab))
	fm_laplacian_moy = int(np.mean(focus_laplacian_tab))
	fm_canny_moy = int(np.mean(focus_canny_tab))
	fm_entropy_moy = int(np.mean(focus_entropy_tab))
	fm_corner_counter_moy = int(np.mean(focus_corner_counter_tab))
	fm_picture_variability_moy = int(np.mean(focus_picture_variability_tab))
	
	return fm_sobel_moy, fm_laplacian_moy, fm_canny_moy, fm_entropy_moy, fm_corner_counter_moy, fm_picture_variability_moy