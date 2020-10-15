#FisherFace - extract the principal component to analyse the facial future vector and compare with testing images
#uses the euclidean distance formula, process complete image at a time

#LBPH - Local Binary Pattern,works with certain point of a image, it will try to find out certain pixel comparing neighbouring pixels
#4 parameters - radius, neighbors, x grid, y grid, in a matrix compare the main spot threshold image with its neighbour and form in a binary pattern.

#block diagram - loading detection algorithm -> loading classifier for face recognition -> training classifier for our dataset
# -> reading frame from camera & pre-processing -> face detection by its algorithm -> predicting face by loading frame into model -> display recognized class with its accuracy


import cv2
import numpy # array based work
import os #path
haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
datasets = 'dataset'
(images, labels, names, id) = ([],[],{},0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
            #print(labels)
        id+=1
(width,height) = (130,100)
(images,labels) = [numpy.array(lis) for lis in [images,labels]]
print(images,labels)

model = cv2.face.LBPHFaceRecognizer_create()
#model = cv2.face.FisherFaceRecognizer_create()
model.train(images,labels)
print("Training Completed")

webcam = cv2.VideoCapture(0)
cnt = 0
while True:
    (_,im) = webcam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,255,0),2)
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face,(width,height))

        prediction = model.predict(face_resize)
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),3)
        if prediction[1]<800:
            cv2.putText(im,'%s - %f.0f'% (names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,2 ,(0,0,255))
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt+=1
            cv2.putText(im,'Unknown',(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0))
            if cnt>100:
                print("Unknown Person")
                cv2.imwrite("input.jpg",im)
                cnt = 0;

webcam.release()
cv2.destroyAllWindows()

