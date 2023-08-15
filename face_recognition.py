def faces_train():
     import os
     import cv2 as cv
     import numpy as np
     people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
     DIR = r'Faces/train'
     haar_cascade = cv.CascadeClassifier('haar_face.xml')
     features = []
     labels = []
     def create_train():
       for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = cv.imread(img_path)
            if img_array is None:
                continue
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            for (x,y,w,h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
     create_train()
     print('-------------Training done ---------------')
     features = np.array(features, dtype='object')
     labels = np.array(labels) 
     face_recognizer = cv.face.LBPHFaceRecognizer_create()
     face_recognizer.train(features,labels)
     face_recognizer.save('face_trained.yml')
     np.save('features.npy', features)
     np.save('labels.npy', labels)

def recognise_face():
     import numpy as np
     import cv2 as cv
     haar_cascade = cv.CascadeClassifier('haar_face.xml')
     people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
     face_recognizer = cv.face.LBPHFaceRecognizer_create()
     face_recognizer.read('face_trained.yml')
     img = cv.imread(r'Faces/val/elton_john/4.jpg')
     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     cv.imshow('Person', gray)
     faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
     for (x,y,w,h) in faces_rect:
       faces_roi = gray[y:y+h,x:x+w]
     label, confidence = face_recognizer.predict(faces_roi)
     print(f'Label = {people[label]} with a confidence of {confidence}')
     cv.putText(img, str(people[label]), (20,20), cv.FONT_ITALIC, 1.0, (0,255,0), thickness=2)
     cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)
     cv.imshow('Detected Face', img)
     cv.waitKey(0)