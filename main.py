import cv2
import matplotlib.pyplot as plt

#step 1 read image using opencv
original_img = cv2.imread('C:/Users/muham/Desktop/IMG_20200818_092132.jpg')
RGB_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
plt.imshow(RGB_img)
plt.show()

#step 2 conver image to gray scal
grayScaleImage = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
plt.imshow(grayScaleImage, cmap='gray')
#plt.show()


#step3 smoothing grayscal
smoothingImage = cv2.medianBlur(grayScaleImage,5)
plt.imshow(smoothingImage, cmap='gray')
#plt.show()


#step4 Extract edges
edges = cv2.adaptiveThreshold(smoothingImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
plt.imshow(edges , cmap='gray')
plt.show()

#step5 preparing a mask image
colorImage = cv2.bilateralFilter(RGB_img, 15, 150, 150)
plt.imshow(colorImage)
plt.show()
#step6 cartoon image
cartoon = cv2.bitwise_and(colorImage,colorImage,mask=edges)
plt.imshow(cartoon)
plt.show()
colorImage = cv2.bilateralFilter(cartoon, 9, 300, 300)
plt.imshow(colorImage)
plt.show()