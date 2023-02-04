import function as fct
import numpy as np
import matplotlib.pyplot as plt
import cv2


########INPUTS################################################################################
Vmin_moteur_autofocus = 0
Vmax_moteur_autofocus = 50
pas_focus = 10

color_frame = (0, 255, 0) #couleur du texte affiché sur l'image (vert)
taille_text_frame = 0.5 #taille du texte affiché sur l'image



cap = cv2.VideoCapture(0)

nbre_image_moy_mesure = 20 #chaque résultat de mesure est la moyenne de nbre_image_moy_mesure images

Position_tourelle = [[0, 0], [0, 45], [45, 45], [45, 90], [90, 90]]#[[posX1, posy1], [posX2, posY2], [posXn, posYn], [...]]

########MAIN################################################################################

for pos_T in Position_tourelle:

	print("---------------------")
	print("position tourelle demandée  : ", pos_T[0],";", pos_T[1])


	best_focus_by_pos = []
	bestnbr_symbole_by_pos = []
	best_angle_by_pos = []
	best_box_by_pos = []
	best_mean_X_by_pos = []
	best_mean_Y_by_pos = []

	
	for pos_M in range(Vmin_moteur_autofocus, Vmax_moteur_autofocus + pas_focus, pas_focus): #Recherche de la position de la mire sur l'image globale
		print("position moteur autofocus  demandée  : ", pos_M)

		nbr_symb_moy, box_mire_moy, mean_X_mire_moy, mean_Y_mire_moy, angle_mire_moy = fct.nbre_symboles_mires_detectes_moyenne(nbre_image_moy_mesure, cap, pos_T, pos_M, color_frame, taille_text_frame) #on calcule le nombre de symboles moyens sur nbre_image_moy_mesure images
		best_focus_by_pos.append(pos_M)
		bestnbr_symbole_by_pos.append(nbr_symb_moy)
		best_angle_by_pos.append(angle_mire_moy)
		best_box_by_pos.append(box_mire_moy)
		best_mean_X_by_pos.append(mean_X_mire_moy)
		best_mean_Y_by_pos.append(mean_Y_mire_moy)
	
	#on cherche la position de la mire qui a le plus grand nombre de symboles
	best_index = bestnbr_symbole_by_pos.index(max(bestnbr_symbole_by_pos))
	print("best index : ", best_index)
	best_focus = best_focus_by_pos[best_index]
	bestnbr_symbole = bestnbr_symbole_by_pos[best_index]
	best_angle = best_angle_by_pos[best_index]
	best_box = best_box_by_pos[best_index]
	best_mean_X = best_mean_X_by_pos[best_index]
	best_mean_Y = best_mean_Y_by_pos[best_index]
	print("best focus : ", best_focus)
	print("best nbr symbole : ", bestnbr_symbole)
	print("best angle : ", best_angle)
	print("best box : ", best_box)
	print("best mean X : ", best_mean_X)
	print("best mean Y : ", best_mean_Y)


	cv2.destroyWindow('Contours & mire') #on ferme la fenêtre des contours

	#ON MET LE MASQUE SUR LA MIRE TROUVEE


	if best_mean_X ==0 or best_mean_Y == 0: #si la mire n'a pas été trouvée, envoie une erreur
		print("erreur : mire non trouvée")
	else:
		

		for pos_M in range(Vmin_moteur_autofocus, Vmax_moteur_autofocus + pas_focus, pas_focus): #Calcul de la valeur de flou uniquement sur l'image de la mire
	
			for i in range(nbre_image_moy_mesure):
				ret, frame = cap.read() #prendre une image
				mire_masque = fct.img_masque_mire(frame, best_box)

				gray_mire = cv2.cvtColor(mire_masque, cv2.COLOR_BGR2GRAY)
				fm_sobel = int(fct.variance_of_image_blur_sobel(gray_mire))
				fm_laplacian = int(fct.variance_of_image_blur_laplacian(gray_mire))
				fm_canny = int(fct.variance_of_image_blur_Canny(gray_mire))


				cv2.putText(frame, "{}{}{}{}".format(" Position de la tourelle :  ",pos_T[0],";", pos_T[1]), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)  
				cv2.putText(frame, "{}".format(" Position de la mire trouvee, calcul du flou"), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)  
				cv2.putText(frame, "{}{}".format(" Position du moteur autofocus :  ",pos_M), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA) 
				cv2.putText(frame, "{}{}".format(" Valeur de flou de la mire avec Sobel : ", fm_sobel), (0, 80), cv2.FONT_HERSHEY_SIMPLEX,taille_text_frame, color_frame, 1, cv2.LINE_AA)
				cv2.putText(frame, "{}{}".format(" Valeur de flou de la mire avec Laplace : ", fm_laplacian), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)
				cv2.putText(frame, "{}{}".format(" Valeur de flou de la mire avec Canny : ", fm_canny), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)
				cv2.putText(frame, "{}{}".format(" Moyenne des valeurs de flou : ", int((fm_sobel+fm_laplacian+fm_canny)/3)), (0, 140), cv2.FONT_HERSHEY_SIMPLEX, taille_text_frame, color_frame, 1, cv2.LINE_AA)


				cv2.imshow('frame', frame) #on affiche l'image de base
				if cv2.waitKey(1) == ord('q'):
					break #si q est appuyé, on quitte la boucle

		cv2.destroyWindow('mire') #on ferme la fenêtre des contours


cap.release()
cv2.destroyAllWindows()


#FAIRE GRAPHES MATPLOTLIB