import numpy as np
import cv2

# Les coordonnées 3D des points de la mire
object_points = np.array([
    [0, 0, 0],   # Coin supérieur gauche
    [1, 0, 0],   # Coin supérieur droit
    [1, 1, 0],   # Coin inférieur droit
    [0, 1, 0]    # Coin inférieur gauche
], dtype=np.float32)

# Les coordonnées 2D des points de la mire dans l'image
image_points = np.array([
    [252, 203],  # Coin supérieur gauche
    [431, 186],  # Coin supérieur droit
    [419, 387],  # Coin inférieur droit
    [233, 399]   # Coin inférieur gauche
], dtype=np.float32)

# Les matrices caractéristiques de la caméra
camera_matrix = np.array([
    [1079.19295, 0, 959.5],
    [0, 1079.19295, 539.5],
    [0, 0, 1]
], dtype=np.float32)

# Les coefficients de distorsion de la caméra
distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Résolution du problème de pose
success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, distortion_coefficients)

# Affichage des résultats
if success:
    print("Vecteur de rotation :\n", rotation_vector)
    print("Vecteur de translation :\n", translation_vector)
else:
    print("La résolution du problème de pose a échoué.")