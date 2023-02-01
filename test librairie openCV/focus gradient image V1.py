# import the necessary packages
from imutils import paths
import argparse
import cv2



#calcule la valeur de flou d'une image

def variance_of_image_blur(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

img_name = 'flower.jpg'
image = cv2.imread(img_name)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fm = variance_of_image_blur(gray)
text = "Value : "
# if the focus measure is less than the supplied threshold,
# then the image should be considered "blurry"

# show the image
cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
cv2.imshow(img_name, image)
print("valeur de flou : ", fm)
key = cv2.waitKey(0)