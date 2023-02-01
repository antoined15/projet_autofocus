import cv2

# Charger l'image
img = cv2.imread("flower.jpg")

# Convertir en niveaux de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer un seuil pour mettre en évidence les contours de la mire
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Trouver les contours de la mire
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Boucle sur les contours pour trouver la mire
for cnt in contours:
    # Si le contour a suffisamment de points
    if len(cnt) >= 5:
        # Calculer les moments
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            # Calculer les coordonnées de la mire
            mire_x = int(M["m10"] / M["m00"])
            mire_y = int(M["m01"] / M["m00"])
            print("La mire est à la position x =", mire_x, "et y =", mire_y)