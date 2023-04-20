import cv2
import numpy as np
import math
import random

############################################################################################################
def distance_min_seuil_intersect(points): #calcule la distance minimale entre un point et un point d'intersection pour que l'intersection soit valide
    max_distance = 0

    for i in range(len(points) - 1):
        distance = math.sqrt((points[i+1][0] - points[i][0])**2 + (points[i+1][1] - points[i][1])**2)
        if distance > max_distance:         # Mettre à jour la plus grande distance si nécessaire
            max_distance = distance
    # Retourner la plus grande distance divisée par 5 --> valeur qui marche bien
    return int(max_distance/8)

############################################################################################################
def test_intersect_valide(points, intersect, distance_min_seuil): #test si l'intersection est valide : si elle est suffisament proche d'un autre point déjà présent

    X = intersect[0]
    Y = intersect[1]
    distance_min = float('inf')
    for point in points: # Calcul de la distance euclidienne entre le point (X, Y) et le point courant
        distance = math.sqrt((X - point[0]) ** 2 + (Y - point[1]) ** 2)
        if distance < distance_min: # Mise à jour de la distance minimale si nécessaire
            distance_min = distance
        
    if distance_min > distance_min_seuil: # Si la distance minimale est inférieure au seuil, l'intersection n'est pas valide
        return False
    return True

############################################################################################################
def intersection(x1, y1, x2, y2, x3, y3, x4, y4):

    # Calculer les coefficients a et b des équations des droites
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3

    det = a1 * b2 - a2 * b1 # Calculer le dénominateur du système d'équations
    if det == 0:     # Si le dénominateur est égal à zéro, les droites sont parallèles
        return None, None

    x = (b2 * c1 - b1 * c2) / det     # Calculer les coordonnées (x, y) du point d'intersection
    y = (a1 * c2 - a2 * c1) / det     # Retourner les coordonnées (x, y) du point d'intersection

    return x, y

############################################################################################################
def rech_intersections(points):

    seuil_test_intersect = distance_min_seuil_intersect(points) #distance minimale entre un point et un point d'intersection pour que l'intersection soit valide
    doublet_points = []
    pts_intersection = []
    for i in range(len(points)): doublet_points.append([points[i], points[(i+1)%len(points)]]) #on crée les doublets de points
        
    for i in range(len(doublet_points)):     #on recherche les intersections entre les doublets de points
        x1 = doublet_points[i][0][0]
        y1 = doublet_points[i][0][1]
        x2 = doublet_points[i][1][0]
        y2 = doublet_points[i][1][1]
        x3 = doublet_points[(i+2)%len(doublet_points)][0][0]
        y3 = doublet_points[(i+2)%len(doublet_points)][0][1]
        x4 = doublet_points[(i+2)%len(doublet_points)][1][0]
        y4 = doublet_points[(i+2)%len(doublet_points)][1][1]
  
        x_int, y_int = intersection(x1, y1, x2, y2, x3, y3, x4, y4) #on calcule les coordonnées du point d'intersection

        if (x_int != None) and (y_int != None): #on teste si l'intersection existe
            if (test_intersect_valide(points, (x_int, y_int), seuil_test_intersect)): #on teste si l'intersection est valide
                pts_intersection.append([int(x_int), int(y_int)])

    return pts_intersection   # liste des points d'intersection

############################################################################################################
def algo_4cotes(points, nbr_pts_polyg_before = 0, img_show = False): #ALGORITHME PRINCIPAL RECURSIF pour diminer le nombre de points du polygone englobant
    #print("nombre de points du polygone résultant : ", points)
    global incr
    global polyg

    if img_show:
        img = np.zeros((400, 400, 3), np.uint8)
    incr +=1
    pts_intersect = rech_intersections(points)

    if img_show:
        for pt in points: cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)     #ffichage de tous les points
        for pt in pts_intersect: cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)     #ffichage des points d'intersection
        for i in range(len(points_init)): cv2.circle(img, (int(points_init[i][0]), int(points_init[i][1])), 3, (255, 0, 100), -1)     #affichage des points initiaux
                
    for pt in pts_intersect: points.append(pt)

    polyg =  cv2.convexHull(np.array(points))
    polyg = [[sublist[0][0], sublist[0][1]] for sublist in polyg]

    long_polyg = len(polyg)

    if img_show:
    #affichage du polygone résultant
        for i in range(long_polyg): cv2.line(img, (int(polyg[i][0]), int(polyg[i][1])), (int(polyg[(i+1)%long_polyg][0]), int(polyg[(i+1)%long_polyg][1])), (255, 255, 255), 1)
        
    #print("nombre de points du polygone résultant : ", long_polyg)
        cv2.imshow('iteration image numero ' + str(incr), img)
        cv2.waitKey()
    
    if long_polyg > 4 :
            if len(polyg) != nbr_pts_polyg_before:
                nbr_pts_polyg_before = long_polyg
                algo_4cotes(polyg,nbr_pts_polyg_before)
                

def approxpoly(points):
    epsilon = 0.01 * cv2.arcLength(np.array(points), True)
    approx = cv2.approxPolyDP(np.array(points), epsilon, True)
    return  [[sublist[0][0], sublist[0][1]] for sublist in approx]

######################################## INPUTS ########################################
global incr #variable globale pour l'incrément de la position des points
global polyg #variable globale pour le polygone, global car sera utilisé par les récursions
incr = 0
global points_init

"""
points = []
points = [[75, 100], [100, 200], [120, 220],[150, 220], [170, 210],  [180, 220], [200, 200], [175, 150], [180, 175], [200, 100]]
#nombre de points aléatoires
#for i in range(100): points.append([random.randint(150, 300), random.randint(150, 300)])

points_init = points.copy() #on garde une copie des points initiaux

 #ALGORITHME
algo_4cotes(points, img_show = False)
polyg_approx = approxpoly(polyg)

######################################## AFFICHAGE DES RESULTATS ########################################
print("polyg : ", polyg_approx)
if len(polyg_approx) != 4:
    print("Pas de polygone englobant de 4 points trouvé")
    print("longeur du polygone englobant : ", len(polyg))

else:

    print("polygone approximé à 4 cotés : ", polyg_approx)

    img = np.zeros((400, 400, 3), np.uint8)
    #afichage des points du polygone approximé
    for i in range(len(polyg_approx)):   cv2.circle(img, (int(polyg_approx[i][0][0]), int(polyg_approx[i][0][1])), 5, (0, 255, 0), -1)  
        
    #affichage despoints initiaux : 
    for i in range(len(points_init)): cv2.circle(img, (int(points_init[i][0]), int(points_init[i][1])), 3, (255, 0, 255), -1)

    cv2.drawContours(img, [polyg_approx], 0, (255, 255, 255), 1) #affichage du polygone approximé   
    cv2.imshow('image approximee', img)

cv2.waitKey(0)
#arrreter quand on appuie sur une touche
cv2.waitKey(1) == ord('q')


"""

