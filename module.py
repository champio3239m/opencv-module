import cv2 as cv
import numpy as np

def oppening_an_image():
    import cv2 as cv
    img=cv.imread('Photos/cat.jpg')
    cv.imshow('Photo',img)
    cv.waitKey(0)

def image_in_grayscale():
    import cv2 as cv
    img=cv.imread('Photos/cat.jpg', cv.IMREAD_GRAYSCALE)
    cv.imshow('Photo',img)
    cv.waitKey(0)

def img_in_HSV():
     import cv2 as cv
     img = cv.imread('Photos/park.jpg')
     hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
     cv.imshow('HSV', hsv)
     cv.waitKey(0)

def oppening_an_video():
    import cv2 as cv
    capture=cv.VideoCapture('Videos/dog.mp4')
    while True:
        isTrue,frame=capture.read()
        cv.imshow('Video',frame)
        if cv.waitKey(20) & 0xFF==('d'):
          break
    capture.release()
    cv.destroyAllWindows()

def resize_an_image():
    import cv2 as cv
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
    import cv2 as cv
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
    import cv2 as cv
    import numpy as np
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
    import cv2 as cv
    import numpy as np
    blank = np.zeros((500,500,3), dtype='uint8')
    cv.imshow ('blank', blank)
    cv.putText(blank, 'insert text here', (0,255), cv.FONT_ITALIC, 1.0,(0,255,0))
    cv.imshow('text', blank)
    cv.waitKey(0)

def blur_gassianblur():
        import cv2 as cv
        img = cv.imread('Photos/cat.jpg')
        blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
        cv.imshow('blured', blur)
        cv.imshow('normal img', img)
        cv.waitKey(0)

def canny_edge():
        import cv2 as cv
        img = cv.imread('Photos/cat.jpg')
        canny = cv.Canny(img, 125,175)
        cv.imshow('canny', canny)
        cv.waitKey(0)

def dilate():
        import cv2 as cv
        img = cv.imread('Photos/cat.jpg')
        canny = cv.Canny(img, 125,175)
        dilated = cv.dilate(canny, (3,3), iterations=1)
        cv.imshow('dilated', dilated)
        cv.waitKey(0)
    
def eroded ():
       import cv2 as cv
       img = cv.imread('Photos/cat.jpg')
       canny = cv.Canny(img, 125,175)
       dilated = cv.dilate(canny, (3,3), iterations=1)
       eroded = cv.erode(dilated, (3,3), iterations=1)
       cv.imshow('eroded pic',eroded)
       cv.waitKey(0)

def resize():
     import cv2 as cv
     import numpy as np
     img = cv.imread('Photos/cat.jpg')
     resize = cv.resize(img, (500,500), interpolation=cv.INTER_AREA)
     cv.imshow('resized', resize)
     cv.waitKey(0)

def cropped():
     import cv2 as cv
     img = cv.imread('Photos/cat.jpg')
     cropped = img[50:200, 200:400]
     cv.imshow('cropped', cropped)
     cv.waitKey(0)

def img_translation():
     import cv2 as cv
     import numpy as np
     img = cv.imread('Photos/cat.jpg', 0)
     rows, cols = img.shape
     M = np.float32([[1,0,100],[0,1,50]])
     dst = cv.warpAffine(img, M, (cols, rows))
     cv.imshow('translated', dst)
     cv.waitKey(0)
     cv.destroyAllWindows()

def img_reflection_horizontal():
     import cv2 as cv
     import numpy as np
     img = cv.imread('Photos/cat.jpg', 0 )
     rows, cols = img.shape
     M = np.float32([[1,0,0], [0,-1, rows], [0,0,1]])
     ref_img = cv.warpPerspective(img, M, (int(cols), int(rows)))
     cv.imshow('reflected horizontally', ref_img)
     cv.waitKey(0)
     cv.destroyAllWindows()

def img_reflection_vertically():
     import cv2 as cv
     import numpy as np
     img = cv.imread('Photos/cat.jpg', 0 )
     rows, cols = img.shape
     M = np.float32([[-1,0,cols], [0,1,0], [0,0,1]])
     ref_img = cv.warpPerspective(img, M, (int(cols), int(rows)))
     cv.imshow('reflected vertically', ref_img)
     cv.waitKey(0)
     cv.destroyAllWindows()

def img_rotation():
     import cv2 as cv
     import numpy as np
     img = cv.imread('Photos/cat.jpg', 0)
     rows, cols = img.shape
     M = np.float32([[1,0,0],[0,-1,rows],[0,0,-1]])
     img_rotated = cv.warpAffine(img, cv.getRotationMatrix2D((cols/2, rows/2),30,0.6), (cols, rows))
     cv.imshow('rotated', img_rotated)
     cv.imwrite('rotated_pic.jpg', img_rotated)
     cv.waitKey(0)     

def shrink_img():
     import numpy as np
     import cv2 as cv
     img = cv.imread('Photos/cat.jpg', 0)
     rows, cols = img.shape
     img_shrinked = cv.resize(img, (250,200), interpolation=cv.INTER_AREA)
     cv.imshow('shrinked', img_shrinked)
     cv.waitKey(0)
     cv.destroyAllWindows()

def enlarge_img():
     import numpy as np
     import cv2 as cv
     img = cv.imread('Photos/cat.jpg', 0)
     rows, cols = img.shape
     img_enlarged = cv.resize(img, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
     cv.imshow('enlarged', img_enlarged)
     cv.waitKey(0)
     cv.destroyAllWindows()

def cropped_img():
     import numpy as np
     import cv2 as cv
     img = cv.imread('Photos/cat.jpg', 0)
     img_cropped = img[100:300, 100:300]
     cv.imwrite('cropped out.jpg', img_cropped)
     cv.waitKey(0)
     cv.destroyAllWindows()

def shearing_x_axis():
     import numpy as np
     import cv2 as cv
     img = cv.imread('Photos/cat.jpg', 0)
     rows, cols = img.shape
     M = np.float32([[1,0.5,0], [0,1,0], [0,0,1]])
     sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
     cv.imshow('img', sheared_img)
     cv.waitKey(0)
     cv.destroyAllWindows()

def shearing_y_axis():
     import cv2 as cv
     import numpy as np
     img = cv.imread('Photos/cat.jpg', 0)
     rows, cols = img.shape
     M = np.float32([[1,0,0],[0.5,1,0],[0,0,1]])
     sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
     cv.imshow('sheared img', sheared_img)
     cv.waitKey(0)
     cv.destroyAllWindows()

def contour_detection():
     import cv2 as cv
     import numpy as np
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
     import cv2 as cv
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
     import cv2 as cv
     img = cv.imread('Photos/lady.jpg')
     cv.imshow('normal photo', img)
     B,G,R = cv.split(img)
     merged = cv.merge([B,G,R])
     cv.imshow("merged", merged)
     cv.waitKey(0)

def averaging_blur():
     import cv2 as cv
     img = cv.imread('Photos/cats.jpg')
     cv.imshow('Cats', img)
     average = cv.blur(img, (3,3))
     cv.imshow('average blur', average)
     cv.waitKey(0)

def median_blur():
     import cv2 as cv
     img = cv.imread('Photos/cats.jpg')
     median = cv.medianBlur(img,3)
     cv.imshow('normal', img)
     cv.imshow('median blur', median)
     cv.waitKey(0)