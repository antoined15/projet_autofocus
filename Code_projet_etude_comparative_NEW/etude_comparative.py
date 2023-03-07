import function as fct
import numpy as np
import matplotlib.pyplot as plt
import cv2
import mplcursors

########INPUTS################################################################################
Vmin_moteur_autofocus = 0
Vmax_moteur_autofocus = 10
pas_focus = 1

color_frame = (0, 255, 0) #couleur du texte affiché sur l'image (vert)
taille_text_frame = 0.5 #taille du texte affiché sur l'image

cap = cv2.VideoCapture(0)

nbre_image_moy_mesure = 10 #chaque résultat de mesure est la moyenne de nbre_image_moy_mesure images
Position_tourelle = [[0, 0], [0, 45]]#, [45, 45]]#, [45, 90], [90, 90]]#[[posX1, posy1], [posX2, posY2], [posXn, posYn], [...]]



########MAIN################################################################################

for pos_T in Position_tourelle:

	print("---------------------")
	fct.position_tourelle(pos_T[0], pos_T[1])


	best_focus_by_pos = []
	bestnbr_symbole_by_pos = []
	best_angle_by_pos = []
	best_box_by_pos = []
	best_mean_X_by_pos = []
	best_mean_Y_by_pos = []

	
	for pos_M in range(Vmin_moteur_autofocus, Vmax_moteur_autofocus + pas_focus, pas_focus): #Recherche de la position de la mire sur l'image globale

		fct.position_moteur_flou(pos_M)

		nbr_symb_moy, box_mire_moy, mean_X_mire_moy, mean_Y_mire_moy, angle_mire_moy = fct.nbre_symboles_mires_detectes_moyenne(nbre_image_moy_mesure, cap, pos_T, pos_M, color_frame, taille_text_frame) #on calcule le nombre de symboles moyens sur nbre_image_moy_mesure images
		best_focus_by_pos.append(pos_M)
		bestnbr_symbole_by_pos.append(nbr_symb_moy)
		best_angle_by_pos.append(angle_mire_moy)
		best_box_by_pos.append(box_mire_moy)
		best_mean_X_by_pos.append(mean_X_mire_moy)
		best_mean_Y_by_pos.append(mean_Y_mire_moy)
	
	#on cherche la position de la mire qui a le plus grand nombre de symboles
	best_index = bestnbr_symbole_by_pos.index(max(bestnbr_symbole_by_pos))
	best_focus = best_focus_by_pos[best_index]
	bestnbr_symbole = bestnbr_symbole_by_pos[best_index]
	best_angle = best_angle_by_pos[best_index]
	best_box = best_box_by_pos[best_index]
	best_mean_X = best_mean_X_by_pos[best_index]
	best_mean_Y = best_mean_Y_by_pos[best_index]

	print("Meilleure position autofocus trouvé : ", best_focus)
	print("Mire en position : ", best_mean_X, ";", best_mean_Y)

	#cv2.destroyWindow('Contours & mire') #on ferme la fenêtre des contours

	#GRAPHES
	fig = plt.figure()
	
	titre = "Recherche mire --> position de la tourelle : " + str(pos_T[0]) + ";" + str(pos_T[1])
	plt.plot(best_focus_by_pos, bestnbr_symbole_by_pos)
	plt.plot(best_focus, bestnbr_symbole, 'ro', label = 'Position du moteur autofocus optimal')
	plt.grid()
	plt.legend(loc = 'lower left')
	plt.xlabel('Position du moteur autofocus')
	plt.ylabel('Nombre de symboles détectés')
	plt.title(titre)
	mplcursors.cursor( hover=True)#, annotations=True, annotation_kwargs=dict(fontsize=10, color='red'))
	plt.show(block = False)


	if best_mean_X ==0 or best_mean_Y == 0: #si la mire n'a pas été trouvée, envoie une erreur
		print("erreur : mire non trouvée")
	else:
		sobel = []
		laplacian = []
		canny = []
		moyenne_fm = []
		posit_M = []

		for pos_M in range(Vmin_moteur_autofocus, Vmax_moteur_autofocus + pas_focus, pas_focus): #Calcul de la valeur de flou uniquement sur l'image de la mire
			
			fm_sobel_moy, fm_laplacian_moy, fm_canny_moy, fm_moyenne_moy = fct.variance_of_image_blur_moyenne(cap,best_box, nbre_image_moy_mesure, color_frame, taille_text_frame, pos_T, pos_M)
			sobel.append(fm_sobel_moy)
			laplacian.append(fm_laplacian_moy)
			canny.append(fm_canny_moy)
			moyenne_fm.append(fm_moyenne_moy)
			posit_M.append(pos_M)

	
		fig = plt.figure()
		#GRAPHES
		titre = "Calcul flou --> position de la tourelle : " + str(pos_T[0]) + ";" + str(pos_T[1])
		plt.plot(posit_M, sobel, label = 'sobel')
		plt.plot(posit_M, laplacian, label = 'laplacian')
		plt.plot(posit_M, canny, label = 'canny')
		plt.plot(posit_M, moyenne_fm, label = 'moyenne')
		plt.legend()
		plt.grid()
		plt.xlabel('Position du moteur autofocus')
		plt.ylabel('Valeur de flou')
		plt.title(titre)
		mplcursors.cursor( hover=True)#, annotations=True, annotation_kwargs=dict(fontsize=10, color='red'))
		plt.show(block = False)

		cv2.destroyWindow('mire') #on ferme la fenêtre des contours

cap.release()
cv2.destroyAllWindows()

plt.show() #pour bloquer les graphes et qu'ils ne disparaissent pas
#FAIRE GRAPHES MATPLOTLIB