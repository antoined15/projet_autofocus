import cv2
import numpy as np
import math
import random


def distance_min_seuil_intersect(points):

    # Initialiser la plus grande distance à 0
    max_distance = 0

    # Parcourir la liste de points
    for i in range(len(points) - 1):
        # Calculer la distance entre les points i et i+1
        distance = math.sqrt((points[i+1][0] - points[i][0])**2 + (points[i+1][1] - points[i][1])**2)
        # Mettre à jour la plus grande distance si nécessaire
        if distance > max_distance:
            max_distance = distance

    # Retourner la plus grande distance divisée par 4
    max_distance = int(max_distance/5)
    #print("distance max = ", max_distance)

    return max_distance


def test_intersect_valide(points, intersect, distance_min_seuil):

    X = intersect[0]
    Y = intersect[1]
    distance_min = float('inf')
    for point in points:
        # Calcul de la distance euclidienne entre le point (X, Y) et le point courant
        distance = math.sqrt((X - point[0]) ** 2 + (Y - point[1]) ** 2)

        # Mise à jour de la distance minimale si nécessaire
        if distance < distance_min:
            distance_min = distance
        
    # Si la distance minimale est inférieure au seuil, l'intersection n'est pas valide
    #print("distance_min = ", distance_min)
    if distance_min > distance_min_seuil:
        return False
    return True


def intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calculer les coefficients a et b des équations des droites
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3
    # Calculer le dénominateur du système d'équations
    det = a1 * b2 - a2 * b1
    # Si le dénominateur est égal à zéro, les droites sont parallèles
    if det == 0:
        return None, None
    # Calculer les coordonnées (x, y) du point d'intersection
    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det
    # Retourner les coordonnées (x, y) du point d'intersection
    return x, y


def rech_intersections(points):

    seuil_test_intersect = distance_min_seuil_intersect(points) #distance minimale entre un point et un point d'intersection pour que l'intersection soit valide
    doublet_points = []
    pts_intersection = []
    for i in range(len(points)):
        doublet_points.append([points[i], points[(i+1)%len(points)]])

    #on recherche les intersections entre les doublets
    for i in range(len(doublet_points)):
        x1 = doublet_points[i][0][0]
        y1 = doublet_points[i][0][1]
        x2 = doublet_points[i][1][0]
        y2 = doublet_points[i][1][1]
        x3 = doublet_points[(i+2)%len(doublet_points)][0][0]
        y3 = doublet_points[(i+2)%len(doublet_points)][0][1]
        x4 = doublet_points[(i+2)%len(doublet_points)][1][0]
        y4 = doublet_points[(i+2)%len(doublet_points)][1][1]
  
        x_int, y_int = intersection(x1, y1, x2, y2, x3, y3, x4, y4)
        if (x_int != None) and (y_int != None):
            if (test_intersect_valide(points, (x_int, y_int), seuil_test_intersect)):
                pts_intersection.append([int(x_int), int(y_int)])


    return pts_intersection   # liste des points d'intersection



def algo_4cotes(points, nbr_pts_polyg_before = 0):
    img = np.zeros((400, 400, 3), np.uint8)
    global incr
    global polyg
    incr +=1
    n_points = len(points)


    

    pts_intersect = rech_intersections(points)


    for pt in pts_intersect:
        #cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        pass

    for pt in pts_intersect:
        points.append(pt)

    polyg =  cv2.convexHull(np.array(points))
    polyg = [[sublist[0][0], sublist[0][1]] for sublist in polyg]
    for i in range(len(polyg)):
        cv2.line(img, (int(polyg[i][0]), int(polyg[i][1])), (int(polyg[(i+1)%len(polyg)][0]), int(polyg[(i+1)%len(polyg)][1])), (255, 255, 255), 1)
        cv2.circle(img, (int(polyg[i][0]), int(polyg[i][1])), 3, (255, 100, 0), -1)
    #dessiner les points entre les lignes successives : 
    for i in range(len(points_init)):
        cv2.circle(img, (int(points_init[i][0]), int(points_init[i][1])), 3, (255, 0, 255), -1)

    print("nombre de points du polygone résultant : ", len(polyg))
    cv2.imshow('img ' + str(incr), img)
    cv2.waitKey()
    
    if len(polyg) > 4 :
            if len(polyg) != nbr_pts_polyg_before:
                nbr_pts_polyg_before = len(polyg)
                algo_4cotes(polyg,nbr_pts_polyg_before)
                

def approxpoly(points):
    epsilon = 0.1 * cv2.arcLength(np.array(points), True)
    approx = cv2.approxPolyDP(np.array(points), epsilon, True)
    return approx




global incr #variable globale pour l'incrément de la position des points
global polyg #variable globale pour le polygone, global car sera utilisé par les récursions
incr = 0
global points_init
points = [[100, 100], [100, 200], [120, 220],[150, 220], [170, 210],  [180, 220], [200, 200], [175, 150], [180, 175], [200, 100]]
points_init = points
#nombre de poits aléatoires
n = 100
##points = []
#for i in range(n):
    #points.append([random.randint(150, 300), random.randint(150, 300)])

algo_4cotes(points)
print("polyg : ", polyg)
polyg_approx = approxpoly(polyg)

print("nombre de points du polygone approximé : ", len(polyg_approx))#nombre de points du polygone approximé
print("polyg approx : ", polyg_approx)


#affichage du polygone approximé
img = np.zeros((400, 400, 3), np.uint8)

#affichage despoints initiaux : 
for i in range(len(points_init)):
    cv2.circle(img, (int(points_init[i][0]), int(points_init[i][1])), 3, (255, 0, 255), -1)

cv2.drawContours(img, [polyg_approx], 0, (0, 0, 255), 1)

#afichage des points du polygone approximé
for i in range(len(polyg_approx)):
    cv2.circle(img, (int(polyg_approx[i][0][0]), int(polyg_approx[i][0][1])), 3, (0, 255, 0), -1)

    

cv2.imshow('img approx', img)

cv2.waitKey(0)
#arrreter quand on appuie sur une touche
cv2.waitKey(1) == ord('q')




