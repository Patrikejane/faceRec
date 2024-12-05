# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 03:08:38 2024

@author: sskma
"""

import cv2
import face_recognition

modi_image = face_recognition.load_image_file('samples/modi.jpg')
modi_face_encodeing = face_recognition.face_encodings(modi_image)[0]


trump_image = face_recognition.load_image_file('samples/trump.jpg')
trump_face_encodeing = face_recognition.face_encodings(trump_image)[0]

era_image = face_recognition.load_image_file('samples/erandi.jpg')
era_face_encodeing = face_recognition.face_encodings(modi_image)[0]


sunimal_image = face_recognition.load_image_file('samples/sunimal.jpg')
sunimal_face_encodeing = face_recognition.face_encodings(trump_image)[0]


known_face_encording = [modi_face_encodeing,trump_face_encodeing,era_face_encodeing,sunimal_face_encodeing]

known_face_namese = ["modi","trump","erandi","sunimal"]


image_to_recognize = face_recognition.load_image_file('testImages/test3.jpg')

all_face_location = face_recognition.face_locations(image_to_recognize,model="hog")

all_face_encording = face_recognition.face_encodings(image_to_recognize,all_face_location)

print('there are {} no of faces in this image'.format(len(all_face_location)))

for current_face_location,current_face_encording in zip(all_face_location,all_face_encording):
    
    top,right,botoom,left = current_face_location

    all_matchers = face_recognition.compare_faces(known_face_encording, current_face_encording)
    
    name = "Unkown Face"
    
    if True in all_matchers:
        first_match_index = all_matchers.index(True)
        name = known_face_namese[first_match_index]
    
    cv2.rectangle(image_to_recognize, (left,top),(right,botoom), (255,0,0),2)   
    
    font = cv2.FONT_HERSHEY_DUPLEX
    
    cv2.putText(image_to_recognize, name, (left,botoom), font, 0.5, (255,255,255),1)
    
    
cv2.imshow("face identified",image_to_recognize)
cv2.waitKey(0)
cv2.destroyAllWindows()