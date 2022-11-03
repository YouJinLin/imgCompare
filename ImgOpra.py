import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    ORI = cv2.imread('Image1.jpg')
    COM = cv2.imread('Image2.jpg')
    ORI = cv2.resize(ORI, (400, 400))
    COM = cv2.resize(COM, (400, 400))
    
    imgORI = cv2.cvtColor(ORI, cv2.COLOR_BGR2GRAY)
    imgCOM = cv2.cvtColor(COM, cv2.COLOR_BGR2GRAY)
    sub = cv2.subtract(imgORI, imgCOM)
    ret, subBinary = cv2.threshold(sub, 100, 200, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), dtype=np.uint8)
    erosion = cv2.erode(subBinary, kernel, iterations=1)
    kernel = np.ones((15, 5), dtype=np.uint8)
    dilation = cv2.dilate(erosion, kernel, iterations=3)
    subBinary = dilation.copy()

    contours, heirachy = cv2.findContours(subBinary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        #cv2.drawContours(ORI, contour, -1, (0, 0, 255), 1)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(ORI, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(ORI, "different", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,  (0, 0, 255), 3)

    plt.subplot(1, 2, 1)
    ORI = cv2.cvtColor(ORI, cv2.COLOR_BGR2RGB) 
    plt.imshow(ORI)
    plt.title('ORI')
    plt.subplot(1, 2, 2)
    COM = cv2.cvtColor(COM, cv2.COLOR_BGR2RGB)
    plt.imshow(COM)
    plt.title('COM')

    plt.show()

    

if __name__ == "__main__":
    main()