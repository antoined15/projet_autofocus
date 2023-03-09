import cv2
import numpy as np

# Nombre de points à générer
n_points = 5

# Générer des points aléatoires
points = np.random.randint(0, 500, size=(n_points, 2))

# Créer une image noire
img = np.zeros((500, 500, 3), np.uint8)

# Dessiner les lignes entre les points successifs
for i in range(n_points - 1):
    pt1 = tuple(points[i])
    pt2 = tuple(points[i + 1])
    cv2.line(img, pt1, pt2, (0, 0, 255), 2)

# Dessiner la ligne reliant le dernier point au premier point
pt1 = tuple(points[-1])
pt2 = tuple(points[0])
cv2.line(img, pt1, pt2, (0, 0, 255), 2)
for pts in points:
    cv2.circle(img, tuple(pts), 5, (0, 255, 0), 2)

# Afficher l'image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
