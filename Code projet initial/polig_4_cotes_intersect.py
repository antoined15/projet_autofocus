import cv2
import numpy as np
import math

def test_intersect_valide(points, intersect):


    distance_min_seuil = 20 #distance minimale entre un point et un point d'intersection pour que l'intersection soit valide
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
    print("distance_min = ", distance_min)
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
        return None
    # Calculer les coordonnées (x, y) du point d'intersection
    x = (b2 * c1 - b1 * c2) / det
    y = (a1 * c2 - a2 * c1) / det
    # Retourner les coordonnées (x, y) du point d'intersection
    return x, y


def rech_intersections(points):

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

        if (test_intersect_valide(points, (x_int, y_int))):
            pts_intersection.append([x_int, y_int])

        #test pour savoir si on garde les points d'intersection : si ils ont incohérents et ne ressemblent pas à des coins de la mire

    print("longueur des points d'intersection : ", len(pts_intersection))

    return pts_intersection   # liste des points d'intersection


img = np.zeros((500, 500, 3), np.uint8)

points = np.array([[100, 105], [80, 200], [200, 200],  [220, 190] ,[220, 110],  [200, 103]])
n_points = len(points)
pts_intersect = rech_intersections(points)


#dessiner les points entre les lignes successives : 
for i in range(n_points):
    cv2.line(img, (int(points[i][0]), int(points[i][1])), (int(points[(i+1)%n_points][0]), int(points[(i+1)%n_points][1])), (0, 255, 0), 1)
    cv2.circle(img, (int(points[i][0]), int(points[i][1])), 3, (255, 0, 255), -1)

#dessinner les points d'intersection
for pts in pts_intersect:
    cv2.circle(img, (int(pts[0]), int(pts[1])), 3, (0, 0, 255), -1)

cv2.imshow('img', img)
cv2.waitKey()


