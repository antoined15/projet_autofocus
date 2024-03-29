import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import os
import cv2
import csv
import argparse

import etude_comparative_fct as fct
from Focuser import Focuser
from picamera2 import Picamera2
from pantilt import Pantilt


########INPUTS################################################################################
color_frame = (0, 255, 0) #couleur du texte affiché sur l'image (vert)
taille_text_frame = 0.5 #taille du texte affiché sur l'image

Position_tourelle = [[90, 13], [105, 35], [90, 60]]#, [45, 45], [45, 90], [90, 90]]#[[posX1, posy1], [posX2, posY2], [posXn, posYn], [...]]

##############################################################################################

camera64mp_max_focus_step = 1023
camera16mp_max_focus_step = 4095

#Initialisaiton

parser = argparse.ArgumentParser(description="Programme pour l'etude comparative des methodes de calculs de focus")

parser.add_argument("-d","--device", type=str, default="64mp" ,choices=["16mp","64mp"], required=True, help="Camera utilise [64mp] or [16mp]")
parser.add_argument("-o","--output-dir", type=str, default="data", help="Dossier de destination pour les resultats")
parser.add_argument("--focus-step", type=int, default=50,help="Incrementation du pas du moteur de focus")
parser.add_argument("--fine-focus-step", type=int, default=5, help="Incrementation du pas du moteur de focus en mode fin")
parser.add_argument("--img-mean", type=int, default=1,help="Nombre d'acquisition d'image moyenne")
parser.add_argument("--fine-img-mean", type=int, default=5, help="Nombre d'acquisition d'images moyenne en mode fin")
parser.add_argument("--delta-fine", type=int, default=50, help="Intervalle de pas du moteur mode fin autour de la position optimale")
parser.add_argument("--width", type=int, default=1280, help="Largeur de l'image")
parser.add_argument("--height", type=int, default=720, help="Hauteur de l'image")


args = parser.parse_args()

try:
	os.system("mkdir "+args.output_dir)
except:
	print(f"Le dossier {args.output_dir} existe deja ou ne peut pas etre creer")

if args.device == "64mp":
	Vmin_moteur_autofocus = 0
	Vmax_moteur_autofocus = camera64mp_max_focus_step
elif args.device == "16mp":
	Vmin_moteur_autofocus = 0
	Vmax_moteur_autofocus = camera16mp_max_focus_step
else :
	Vmin_moteur_autofocus = 0
	Vmax_moteur_autofocus = 1000

if args.device == "64mp" or args.device == "16mp":
	picam2 = Picamera2()
	#picam2.configure(picam2.create_preview_configuration(main={"format":"RGB888","size":(args.width,args.height)}, lores={"size":(800,600)}, display="lores"))
	picam2.configure(picam2.create_preview_configuration(main={"format":"RGB888","size":(args.width,args.height)}))
	picam2.start()
	focuser = Focuser("/dev/v4l-subdev1")
	focuser.step = 0

print(f"Type de caméra : {args.device}\tVmin_moteur_autofocus = {Vmin_moteur_autofocus}\tVmax_moteur_autofocus = {Vmax_moteur_autofocus}")	

# Initialisation de la tourelle pan-tilt
pt = Pantilt()

if pt.connect() == -1:
	print("Impossible de connecter la tourelle pan-tilt")
	exit()

########MAIN################################################################################

for pos_T in Position_tourelle:

	#print("---------------------")
	fct.position_tourelle(pt,pos_T[0], pos_T[1])

	best_focus_by_pos = []
	bestnbr_symbole_by_pos = []
	best_angle_by_pos = []
	best_box_by_pos = []
	best_mean_X_by_pos = []
	best_mean_Y_by_pos = []

	
	for pos_M in range(Vmin_moteur_autofocus, Vmax_moteur_autofocus + args.focus_step, args.focus_step): #Recherche de la position de la mire sur l'image globale

		fct.position_moteur_flou(focuser, args.device, pos_M)

		nbr_symb_moy, box_mire_moy, mean_X_mire_moy, mean_Y_mire_moy, angle_mire_moy = fct.nbre_symboles_mires_detectes_moyenne(args.img_mean, picam2, pos_T, pos_M, color_frame, taille_text_frame) #on calcule le nombre de symboles moyens sur nbre_image_moy_mesure images
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

	for pos_M in range(best_focus-args.delta_fine,best_focus+args.delta_fine,args.fine_focus_step):
		fct.position_moteur_flou(focuser, args.device, pos_M)

		nbr_symb_moy, box_mire_moy, mean_X_mire_moy, mean_Y_mire_moy, angle_mire_moy = fct.nbre_symboles_mires_detectes_moyenne(args.fine_img_mean, picam2, pos_T, pos_M, color_frame, taille_text_frame) #on calcule le nombre de symboles moyens sur nbre_image_moy_mesure images
		best_focus_by_pos.append(pos_M)
		bestnbr_symbole_by_pos.append(nbr_symb_moy)
		best_angle_by_pos.append(angle_mire_moy)
		best_box_by_pos.append(box_mire_moy)
		best_mean_X_by_pos.append(mean_X_mire_moy)
		best_mean_Y_by_pos.append(mean_Y_mire_moy)

	best_focus_by_pos,bestnbr_symbole_by_pos,best_angle_by_pos,best_box_by_pos,best_mean_X_by_pos,best_mean_Y_by_pos = fct.trie_liste(best_focus_by_pos,bestnbr_symbole_by_pos,best_angle_by_pos,best_box_by_pos,best_mean_X_by_pos,best_mean_Y_by_pos)

	best_index = bestnbr_symbole_by_pos.index(max(bestnbr_symbole_by_pos))
	best_focus = best_focus_by_pos[best_index]
	bestnbr_symbole = bestnbr_symbole_by_pos[best_index]
	best_angle = best_angle_by_pos[best_index]
	best_box = best_box_by_pos[best_index]
	best_mean_X = best_mean_X_by_pos[best_index]
	best_mean_Y = best_mean_Y_by_pos[best_index]

	data = list(zip(best_focus_by_pos,bestnbr_symbole_by_pos,best_angle_by_pos,best_box_by_pos,best_mean_X_by_pos,best_mean_Y_by_pos))
	header = ["best_focus_by_pos","bestnbr_symbole_by_pos","best_angle_by_pos","best_box_by_pos","best_mean_X_by_pos","best_mean_Y_by_pos"]

	# Save in csv
	with open(f'data/symbole_step({pos_T[0]},{pos_T[1]}).csv','w', newline='', encoding="utf-8") as csv_file:
		csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"',quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(header)
		csv_writer.writerows(data)

	fig = plt.figure()
	
	titre = "Recherche mire --> position de la tourelle : " + str(pos_T[0]) + ";" + str(pos_T[1])
	plt.plot(best_focus_by_pos, bestnbr_symbole_by_pos, '+-')
	plt.plot(best_focus, bestnbr_symbole, 'ro', label = 'Position du moteur autofocus optimal')
	plt.grid()
	plt.legend(loc = 'lower left')
	plt.xlabel('Position du moteur autofocus')
	plt.ylabel('Nombre de symboles détectés')
	plt.title(titre)	#plt.show(block = False)


	if best_mean_X ==0 or best_mean_Y == 0: #si la mire n'a pas été trouvée, envoie une erreur
		print("erreur : mire non trouvée")
		pass
	else:
		sobel = []
		laplacian = []
		canny = []
		picture_entropy = []
		corner_counter = []
		picture_variability = []
		posit_M = []

		for pos_M in range(Vmin_moteur_autofocus, Vmax_moteur_autofocus + args.focus_step, args.focus_step): #Calcul de la valeur de flou uniquement sur l'image de la mire
			
			fct.position_moteur_flou(focuser, args.device, pos_M)

			fm_sobel_moy, fm_laplacian_moy, fm_canny_moy, fm_entropy_moy, fm_corner_counter_moy, fm_picture_variability_moy = fct.variance_of_image_blur_moyenne(picam2,best_box, args.fine_img_mean, color_frame, taille_text_frame, pos_T, pos_M)
			sobel.append(fm_sobel_moy)
			laplacian.append(fm_laplacian_moy)
			canny.append(fm_canny_moy)
			picture_entropy.append(fm_entropy_moy)
			corner_counter.append(fm_corner_counter_moy)
			picture_variability.append(fm_picture_variability_moy)
			posit_M.append(pos_M)

		for pos_M in range(best_focus-args.delta_fine,best_focus+args.delta_fine,args.fine_focus_step):
			fct.position_moteur_flou(focuser, args.device, pos_M)

			fm_sobel_moy, fm_laplacian_moy, fm_canny_moy, fm_entropy_moy, fm_corner_counter_moy, fm_picture_variability_moy = fct.variance_of_image_blur_moyenne(picam2,best_box, args.fine_img_mean, color_frame, taille_text_frame, pos_T, pos_M)
			sobel.append(fm_sobel_moy)
			laplacian.append(fm_laplacian_moy)
			canny.append(fm_canny_moy)
			picture_entropy.append(fm_entropy_moy)
			corner_counter.append(fm_corner_counter_moy)
			picture_variability.append(fm_picture_variability_moy)
			posit_M.append(pos_M)

		posit_M,laplacian,canny,picture_entropy,corner_counter,picture_variability,sobel = fct.trie_liste(posit_M,laplacian,canny,picture_entropy,corner_counter,picture_variability,sobel)

		data = list(zip(posit_M,laplacian,canny,picture_entropy,corner_counter,picture_variability,sobel))
		header = ["pos_m","laplacian","canny","entropy","corner_counter","sobel"]

		with open(f'data/method_step({pos_T[0]},{pos_T[1]}).csv','w', newline='', encoding="utf-8") as csv_file:
			csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"',quoting=csv.QUOTE_MINIMAL)
			csv_writer.writerow(header)
			csv_writer.writerows(data)

		# Save in csv

		fig = plt.figure()
		#GRAPHES
		titre = "Calcul flou --> position de la tourelle : " + str(pos_T[0]) + ";" + str(pos_T[1])
		plt.plot(posit_M, sobel,'+-',  label = 'sobel', )
		plt.plot(posit_M, laplacian,'+-', label = 'laplacian')
		plt.plot(posit_M, canny,'+-', label = 'canny')
		plt.plot(posit_M, picture_entropy,'+-', label = 'entropie image')
		plt.plot(posit_M, corner_counter,'+-', label = 'coins détectés')
		plt.plot(posit_M, picture_variability,'+-', label = 'variabilité image')
		plt.legend()
		plt.grid()
		plt.xlabel('Position du moteur autofocus')
		plt.ylabel('Valeur de flou')
		plt.yscale('log')
		plt.title(titre)
		mplcursors.cursor(hover=True)
		plt.show(block = False)

		cv2.destroyWindow('mire') #on ferme la fenêtre des contours

picam2.stop()
cv2.destroyAllWindows()

plt.show(block = True) #pour bloquer les graphes et qu'ils ne disparaissent pas
#FAIRE GRAPHES MATPLOTLIB