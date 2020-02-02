import cv2
import numpy as np
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

_, frame = cap.read()
rows, cols, _ = frame.shape
pic_mask = np.zeros((rows, cols), np.uint8)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")

glass = cv2.imread("img/face2.png")
face_effect = cv2.imread("img/face1.png")
nose = cv2.imread("img/nose2.png")


top = 21
center = 27
left = 36
right = 45
height_ratio = 0.46
scall = 1.7
img = glass.copy()


def menu():
    print("\n###########################\n\
     Demonstration loop \n \
        you can press keys: \n\
    \t q: quit \n\
    \t g: Glass\n\
    \t f: Face \n\
    \t n: Nose \n")

menu()
while True:
    _, frame = cap.read()
    pic_mask.fill(0)
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(frame)
    if faces is not None:
        for face in faces:
            landmarks = predictor(gray_frame, face)
            
            if landmarks is not None:
                top_position = (landmarks.part(top).x, landmarks.part(top).y)
                center_position = (landmarks.part(center).x, landmarks.part(center).y)
                left_position = (landmarks.part(left).x, landmarks.part(left).y)
                right_position = (landmarks.part(right).x, landmarks.part(right).y)
                
                obj_width = int(hypot(left_position[0] - right_position[0],
                                           left_position[1] - right_position[1]) * scall)
                obj_height = int(obj_width * height_ratio)
                
                top_left = (int(center_position[0] - obj_width / 2),
                                              int(center_position[1] - obj_height / 2))
                bottom_right = (int(center_position[0] + obj_width / 2),
                                       int(center_position[1] + obj_height / 2))
                
                
                effect_pic = cv2.resize(img, (obj_width, obj_height))
                effect_pic_gray = cv2.cvtColor(effect_pic, cv2.COLOR_BGR2GRAY)
                _, effect_mask = cv2.threshold(effect_pic_gray, 25, 255, cv2.THRESH_BINARY_INV)
                
                effect_area = frame[top_left[1]: top_left[1] + obj_height,
                                    top_left[0]: top_left[0] + obj_width]
                effect_area_no_obj = cv2.bitwise_and(effect_area, effect_area, mask=effect_mask)
                final_effect = cv2.add(effect_area_no_obj, effect_pic)
                
                frame[top_left[1]: top_left[1] + obj_height,
                                    top_left[0]: top_left[0] + obj_width] = final_effect

    
        cv2.imshow("Result", frame)
    
    key = cv2.waitKey(1)
    if  key == ord('g'):
        top = 21
        center = 27
        left = 36
        right = 45
        height_ratio = 0.46
        scall = 1.7
        img = glass.copy()
        
        menu()
        
    if key == ord('f'):
        top = 19
        center = 28
        left = 36
        right = 45
        height_ratio = 0.70
        scall = 1.4
        img = face_effect.copy()
        
        menu()
    
    if key == ord('n'):
        top = 29
        center = 30
        left = 31
        right = 35
        height_ratio = 1
        scall = 2.3
        img = nose.copy()
        
        menu()
            
    if key == ord('q'):
        break
