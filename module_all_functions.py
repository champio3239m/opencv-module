import cv2 as cv
import numpy as np
import os

def oppening_an_image():
    img=cv.imread('Photos/cat.jpg')
    cv.imshow('Photo',img)
    cv.waitKey(0)

def image_in_grayscale():
    img=cv.imread('Photos/cat.jpg', cv.IMREAD_GRAYSCALE)
    cv.imshow('Photo',img)
    cv.waitKey(0)

def img_in_HSV():
     img = cv.imread('Photos/park.jpg')
     hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
     cv.imshow('HSV', hsv)
     cv.waitKey(0)

def oppening_an_video():
    capture=cv.VideoCapture('Videos/dog.mp4')
    while True:
        isTrue,frame=capture.read()
        cv.imshow('Video',frame)
        if cv.waitKey(20) & 0xFF==('d'):
          break
    capture.release()
    cv.destroyAllWindows()

def resize_an_image():
    img=cv.imread('Photos/cat.jpg')
    cv.imshow('Cat',img)
    def rescaleFrame(frame,scale=0.75):
        width=int(frame.shape [1]*scale)
        height=int(frame.shape [0]*scale)
        dimensions=(width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
    resized_image=rescaleFrame(img)
    cv.imshow('Resized Cat',resized_image)
    cv.waitKey(0)

def resizing_an_video():
    capture=cv.VideoCapture('Videos/dog.mp4')
    def rescaleFrame(frame,scale=0.75):
        width=int(frame.shape [1]*scale)
        height=int(frame.shape [0]*scale)
        dimensions=(width,height)
        return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

    while True:
       isTrue,frame=capture.read()
       cv.imshow('Dog',frame)
       frame_resized=rescaleFrame(frame)
       cv.imshow('Resized Dog',frame_resized)
       if cv.waitKey(20)&0xFF==('d'):
          break
    capture.release()
    cv.destroyAllWindows()

def shapessss():
    blank = np.zeros((500,500,3), dtype='uint8')
    cv.imshow ('blank', blank)
    blank[200:300, 300:400] = 0,255,0
    cv.imshow('green', blank)

    cv.rectangle(blank, (0,0),(blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=-1)
    cv.imshow('recrangle', blank)

    cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,255,0), thickness=3)
    cv.imshow('circle', blank)

    cv.line(blank,(0,0),(300,400),(255,255,255), thickness=3)
    cv.imshow('line', blank)

    cv.waitKey(0)

def text():
    blank = np.zeros((500,500,3), dtype='uint8')
    cv.imshow ('blank', blank)
    cv.putText(blank, 'insert text here', (0,255), cv.FONT_ITALIC, 1.0,(0,255,0))
    cv.imshow('text', blank)
    cv.waitKey(0)

def blur_gassianblur():
        img = cv.imread('Photos/cat.jpg')
        blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
        cv.imshow('blured', blur)
        cv.imshow('normal img', img)
        cv.waitKey(0)

def dilate():
        img = cv.imread('Photos/cat.jpg')
        canny = cv.Canny(img, 125,175)
        dilated = cv.dilate(canny, (3,3), iterations=1)
        cv.imshow('dilated', dilated)
        cv.waitKey(0)
    
def eroded ():
       img = cv.imread('Photos/cat.jpg')
       canny = cv.Canny(img, 125,175)
       dilated = cv.dilate(canny, (3,3), iterations=1)
       eroded = cv.erode(dilated, (3,3), iterations=1)
       cv.imshow('eroded pic',eroded)
       cv.waitKey(0)

def resize():
     img = cv.imread('Photos/cat.jpg')
     resize = cv.resize(img, (500,500), interpolation=cv.INTER_AREA)
     cv.imshow('resized', resize)
     cv.waitKey(0)

def cropped():
     img = cv.imread('Photos/cat.jpg')
     cropped = img[50:200, 200:400]
     cv.imshow('cropped', cropped)
     cv.waitKey(0)

def img_translation():
     img = cv.imread('Photos/cat.jpg', 0)
     rows, cols = img.shape
     M = np.float32([[1,0,100],[0,1,50]])
     dst = cv.warpAffine(img, M, (cols, rows))
     cv.imshow('translated', dst)
     cv.waitKey(0)
     cv.destroyAllWindows()

def img_reflection_horizontal():
     img = cv.imread('Photos/cat.jpg', 0 )
     rows, cols = img.shape
     M = np.float32([[1,0,0], [0,-1, rows], [0,0,1]])
     ref_img = cv.warpPerspective(img, M, (int(cols), int(rows)))
     cv.imshow('reflected horizontally', ref_img)
     cv.waitKey(0)
     cv.destroyAllWindows()

def img_reflection_vertically():
     img = cv.imread('Photos/cat.jpg', 0 )
     rows, cols = img.shape
     M = np.float32([[-1,0,cols], [0,1,0], [0,0,1]])
     ref_img = cv.warpPerspective(img, M, (int(cols), int(rows)))
     cv.imshow('reflected vertically', ref_img)
     cv.waitKey(0)
     cv.destroyAllWindows()

def img_rotation():
     img = cv.imread('Photos/cat.jpg', 0)
     rows, cols = img.shape
     M = np.float32([[1,0,0],[0,-1,rows],[0,0,-1]])
     img_rotated = cv.warpAffine(img, cv.getRotationMatrix2D((cols/2, rows/2),30,0.6), (cols, rows))
     cv.imshow('rotated', img_rotated)
     cv.imwrite('rotated_pic.jpg', img_rotated)
     cv.waitKey(0)     

def shrink_img():
     img = cv.imread('Photos/cat.jpg', 0)
     rows, cols = img.shape
     img_shrinked = cv.resize(img, (250,200), interpolation=cv.INTER_AREA)
     cv.imshow('shrinked', img_shrinked)
     cv.waitKey(0)
     cv.destroyAllWindows()

def enlarge_img():
     img = cv.imread('Photos/cat.jpg', 0)
     rows, cols = img.shape
     img_enlarged = cv.resize(img, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
     cv.imshow('enlarged', img_enlarged)
     cv.waitKey(0)
     cv.destroyAllWindows()

def cropped_img():
     img = cv.imread('Photos/cat.jpg', 0)
     img_cropped = img[100:300, 100:300]
     cv.imwrite('cropped out.jpg', img_cropped)
     cv.waitKey(0)
     cv.destroyAllWindows()

def shearing_x_axis():
     img = cv.imread('Photos/cat.jpg', 0)
     rows, cols = img.shape
     M = np.float32([[1,0.5,0], [0,1,0], [0,0,1]])
     sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
     cv.imshow('img', sheared_img)
     cv.waitKey(0)
     cv.destroyAllWindows()

def shearing_y_axis():
     img = cv.imread('Photos/cat.jpg', 0)
     rows, cols = img.shape
     M = np.float32([[1,0,0],[0.5,1,0],[0,0,1]])
     sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
     cv.imshow('sheared img', sheared_img)
     cv.waitKey(0)
     cv.destroyAllWindows()

def contour_detection():
     img = cv.imread('Photos/cat.jpg')
     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     edged = cv.Canny(gray, 30,300)
     contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
     cv.imshow('canny edges after contouring', edged)
     print('number of contours ='+ str(len(contours)))
     cv.drawContours(img, contours, -1, (0,255,0),3)
     cv.imshow('contours', img)
     cv.waitKey(0)
     cv.destroyAllWindows()

def BGR_colorSpace_split():
     img = cv.imread('Photos/park.jpg')
     B,G,R = cv.split(img)
     cv.imshow('Original', img)
     cv.waitKey(0)
     cv.imshow('Blue', B)
     cv.waitKey(0)
     cv.imshow('Green', G)
     cv.waitKey(0)
     cv.imshow('Red', R)
     cv.waitKey(0)
     cv.destroyAllWindows()

def merge_bgr():
     img = cv.imread('Photos/lady.jpg')
     cv.imshow('normal photo', img)
     B,G,R = cv.split(img)
     merged = cv.merge([B,G,R])
     cv.imshow("merged", merged)
     cv.waitKey(0)

def averaging_blur():
     img = cv.imread('Photos/cats.jpg')
     cv.imshow('Cats', img)
     average = cv.blur(img, (3,3))
     cv.imshow('average blur', average)
     cv.waitKey(0)

def median_blur():
     img = cv.imread('Photos/cats.jpg')
     median = cv.medianBlur(img,3)
     cv.imshow('normal', img)
     cv.imshow('median blur', median)
     cv.waitKey(0)

def bilateral_blur():
     img = cv.imread('Photos/park.jpg')
     cv.imshow('Normal Photo', img)
     biblur = cv.bilateralFilter(img, 10,35,25)
     cv.imshow('BiLateral blur photo', biblur)
     cv.waitKey(0)

def bitwise():
     blank = np.zeros((400,400), dtype='uint8')
     #A UINT8 is an 8-bit unsigned integer (range: 0 through 255 decimal).
     rectangle = cv.rectangle(blank.copy(),(30,30),(370,370), 255, -1)
     # we are giving one parameter for color as it is a binary image, 255 is white
     circle = cv.circle(blank.copy(),(200,200), 200, 255, -1  )
     #200 is the radius 
     cv.imshow('rectangle', rectangle)
     cv.imshow('circle', circle)
     #bitwise AND --> intersecting regions
     bitwise_and= cv.bitwise_and(rectangle, circle)
     # bitwise and took both the images and placed them on top of eachother and
     # returned the common regions
     cv.imshow('Bitwise AND', bitwise_and)
     #bitwise OR --> shows intersecting & non intersecting regions
     bitwise_or = cv.bitwise_or(rectangle, circle)
     # bitwise or took this image and placed it on top and found the common
     # regions and also found uncommon regions and joint them
     cv.imshow('Bitwise OR', bitwise_or)
     #bitwise XOR --> this shows the non intersecting regions
     bitwise_xor = cv.bitwise_xor(rectangle, circle)
     cv.imshow('Bitwise XOR', bitwise_xor)
     # bitwise not --> this inverts the binary color
     bitwise_not_rect = cv.bitwise_not(rectangle)
     cv.imshow('Bitwise NOT', bitwise_not_rect)
     bitwise_not_circ = cv.bitwise_not(circle)
     cv.imshow('Bitwise NOT', bitwise_not_circ)
     cv.waitKey(0)

def masking():
     # masking allows us to focus on certain parts of an image
     img = cv.imread('Photos/cats 2.jpg')
     cv.imshow('Photo', img)
     blank = np.zeros(img.shape[:2], dtype='uint8')
     # dimensions of mask must be of the same size of image
     cv.imshow('blank.', blank)
     mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 135,255,-1)
     cv.imshow('Mask', mask)
     masked = cv.bitwise_and(img, img, mask=mask)
     cv.imshow('Masked', masked)
     cv.waitKey(0)

def histogram_computation_gray():
     import matplotlib.pyplot as plt
     img = cv.imread('Photos/cats.jpg')
     cv.imshow('cats', img)
     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     cv.imshow('gray cats', gray)
     #GrayScale Histogram
     gray_hist = cv.calcHist([gray], [0], None,[256],[0,256])
     plt.figure()
     plt.title('GrayScale Histogram')
     plt.xlabel('Bins')
     plt.ylabel('# of pixels')
     plt.plot(gray_hist)
     plt.xlim([0,256])
     plt.show()
     cv.waitKey(0)

def histogram_computation_bgr():
     import matplotlib.pyplot as plt
     img = cv.imread('Photos/cats.jpg')
     cv.imshow('Cats', img)
     blank = np.zeros(img.shape[:2], dtype='uint8')
     mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)
     masked = cv.bitwise_and(img,img,mask=mask)
     cv.imshow('Mask', masked)
     plt.figure()
     plt.title('Colour Histogram')
     plt.xlabel('Bins')
     plt.ylabel('# of pixels')
     colors = ('b', 'g', 'r')
     for i,col in enumerate(colors):
          hist = cv.calcHist([img], [i], mask, [256], [0,256])
          plt.plot(hist, color=col)
          plt.xlim([0,256])
     plt.show()
     cv.waitKey(0)

def thresholding():
     # convert pics into binary pics, pics are either zero(black) or 255(white)
     img = cv.imread('Photos/cat.jpg')
     cv.imshow('normal pic', img)
     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     cv.imshow('grayscale pic', gray)
     #simple Thresholding
     threshhold, thresh  = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
     #this looks at the image and compares each pixel value to ths threshold value
     # and if it is above this value it set's it to 255, if its below then it sets it to 0
     cv.imshow('threshholded pic', thresh)
     threshhold, thresh_inv  = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
     cv.imshow('inversed thresholded pic', thresh_inv)
     #adaptive thresholding method
     adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
     cv.imshow('Adaptive thresh image', adaptive_thresh)
     cv.waitKey(0)

def edge_detection_laplacian():
     img = cv.imread('Photos/cats.jpg')
     cv.imshow('Cats', img)
     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     cv.imshow('Grayscale Casts', gray)
     lap = cv.Laplacian(gray, cv.CV_64F)
     lap = np.uint8(np.absolute(lap))
     cv.imshow('Laplacian Blur', lap)
     cv.waitKey(0)

def edge_detection_sobel():
     img = cv.imread('Photos/cats.jpg')
     cv.imshow('Cats', img)
     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     cv.imshow('Grayscale Casts', gray)
     sobelx = cv.Sobel(gray, cv.CV_64F,1,0)
     sobely = cv.Sobel(gray, cv.CV_64F,0,1)
     sobel = cv.bitwise_or(sobelx, sobely)
     cv.imshow('Sobel Blur', sobel)
     cv.waitKey(0)

def edge_detection_canny():
        img = cv.imread('Photos/cat.jpg')
        canny = cv.Canny(img, 125,175)
        cv.imshow('canny', canny)
        cv.waitKey(0)

def face_detection():
    img = cv.imread('Photos/group 2.jpg')
    cv.imshow('Person', img)
    # first convert our img to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Grayscale img', gray)
    haar_cascade = cv.CascadeClassifier('haar_face.xml')
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
    print(f'Number of faces found = {len(faces_rect)}')
    for (x,y,w,h) in faces_rect:
        cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
    cv.imshow('Detected faces', img)
    cv.waitKey(0)
