# OpenGL Initialization
def init():
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glEnable(GL_DEPTH_TEST)

# OpenGL Resize Function
def resize(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, float(w)/float(h), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
# OpenGL Render Function
def render():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Set the camera
    gluPerspective(45, 1.33, 0.1, 50.0)
    gluLookAt(0, 0, -5, 0, 0, 0, 0, 1, 0)
    
    # Draw a cube at hand position
    if hand_landmarks:
        for landmark in hand_landmarks.landmark:
            x, y, z = landmark.x, landmark.y, landmark.z
            glTranslatef(x, y, z)
            glutWireCube(1)
            
        glutSwapBuffers()

# OpenGL Display Function
def display():
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                height, width, _ = img.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                
            render(img, hand_landmarks)
    cv2.imshow('Image', img)        
    
# Main Loop    
if __name__ == "__main__":
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutCreateWindow("3D Interaction")

    glutReshapeFunc(resize)
    glutDisplayFunc(display)

    init()

    glutMainLoop()
    cap.release()
    cv2.destroyAllWindows()
        