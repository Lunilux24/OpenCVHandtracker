# Libraries Used: OpenCV, Mediapipe, pyglet
import cv2
import time
import mediapipe as mp
from google.protobuf.json_format import MessageToDict
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from threading import Thread

# Initialize the Model 
mpHands = mp.solutions.hands 
hands = mpHands.Hands( 
	static_image_mode=False, 
	model_complexity=1,
	min_detection_confidence=0.75, 
	min_tracking_confidence=0.75, 
	max_num_hands=2) 

# Initialize Webcam Capture
cap = cv2.VideoCapture(0) # 0 is the default camera

# Initialize OpenGL window dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Initialize canvas
canvas = None
drawing_color = (0, 255, 0)
drawing_thickness = 5

# Initialize time variables
pTime = time.time()  # Initialize pTime here

# OpenGL Initialization
def init():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT)

# OpenGL Drawing Function
def draw():
    glClear(GL_COLOR_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    if canvas is not None:
        glBindTexture(GL_TEXTURE_2D, canvas.id)
        glEnable(GL_TEXTURE_2D)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex2f(0, 0)
        glTexCoord2f(1, 0)
        glVertex2f(WINDOW_WIDTH, 0)
        glTexCoord2f(1, 1)
        glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT)
        glTexCoord2f(0, 1)
        glVertex2f(0, WINDOW_HEIGHT)
        glEnd()
        glDisable(GL_TEXTURE_2D)

# Hand Tracking Function
def hand_tracking():
    global pTime  # Declare pTime as global here
    while True: 
        # Read video frame by frame 
        success, img = cap.read() 
        
        # Flip the image(frame) 
        img = cv2.flip(img, 1) 
        
        # Get the current time
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert BGR image to RGB image 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
        # Process the RGB image 
        results = hands.process(imgRGB) 
        
        # If hands are present in image(frame) 
        if results.multi_hand_landmarks: 
            # Loop through each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image
                for landmark in hand_landmarks.landmark:
                    height, width, _ = img.shape
                    cx, cy = int(landmark.x * width), int(landmark.y * height)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                
                # Connect the landmarks to form an exoskeleton    
                connections = mpHands.HAND_CONNECTIONS
                for connection in connections:
                    point1 = connection[0]
                    point2 = connection[1]
                    x1, y1 = int(hand_landmarks.landmark[point1].x * width), int(hand_landmarks.landmark[point1].y * height)
                    x2, y2 = int(hand_landmarks.landmark[point2].x * width), int(hand_landmarks.landmark[point2].y * height)
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Get the landmarks for specific fingers
                thumb_tip = hand_landmarks.landmark[mpHands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mpHands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP]
                
                # Do something with the landmarks
                

            # Both Hands are present in image(frame) 
            if len(results.multi_handedness) == 2: 
                # Display 'Both Hands' on the image 
                cv2.putText(img, 'Both Hands', (250, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2) 
    
            # If any hand present 
            else: 
                for i in results.multi_handedness: 
                    # Return whether it is Right or Left Hand 
                    label = MessageToDict(i)['classification'][0]['label'] 
    
                    if label == 'Left': 
                        # Display 'Left Hand' on 
                        # left side of window 
                        cv2.putText(img, label+' Hand', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2) 
    
                    if label == 'Right': 
                        # Display 'Left Hand' 
                        # on left side of window 
                        cv2.putText(img, label+' Hand', (460, 50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2) 
    
        # Display Video and when 'q' 
        # is entered, destroy the window 
        cv2.imshow('Image', img) 
        if cv2.waitKey(1) & 0xff == ord('q'): 
            break

# Start hand tracking in a separate thread
thread = Thread(target=hand_tracking)
thread.start()

# Initialize OpenGL window
glutInit()
glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT)
glutCreateWindow("Hand Tracking")

# Register OpenGL functions
glutDisplayFunc(draw)
glutIdleFunc(draw)
init()

# Enter the OpenGL main loop
glutMainLoop()
