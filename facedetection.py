import cv2
import os
import time
img_path = 'img'

face_detector = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')

def get_faces(img_path):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    while True:
     count = 0
     faces = face_detector.detectMultiScale(img_gray, 1.2, 5)
     for (x,y,w,h) in faces :
       cv2.imwrite('img/MinhTuan1_face_{}.jpg'.format(count), img[y:y+h , x : x+w])
       cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
       count += 1
     # cv2.imshow('img', img)

     # if cv2.waitKey(1) & 0xFF == ord('q') :
     # break

     #cv2.destroyAllWindows()
def check_dir(img_path):
 for whatelse in os.listdir(img_path):
    whatelse_path = os.path.join(img_path, whatelse)
    for sub_whatelse in os.listdir(whatelse_path) :
        img_path = os.path.join(whatelse_path, sub_whatelse)
        if img_path.endswith('.jpg'):
            get_faces(img_path)

cam = cv2.VideoCapture('gettyimages-464058554-640_adpp.mp4')
count = 114
while True :
    OK , FRAME = cam.read()
    faces = face_detector.detectMultiScale(FRAME, 1.2, 5)
    for (x,y,w,h) in faces :
        time.sleep(0.25)
        roi = cv2.resize(FRAME[y+2: y+h - 2 , x+2 : x+w - 2], (100,100))
        cv2.imwrite('Datasets/messi/roi_{}.jpg'.format(count), roi)

        cv2.rectangle(FRAME, (x,y), (x+w,y+h), (0,255,0), 2)
        count += 1
    cv2.imshow('cam', FRAME)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

cv2.destroyAllWindows()

