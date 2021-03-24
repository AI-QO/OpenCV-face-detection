# Import OpenCV
import cv2

# Load pre-trained model on face frontals from OpenCV (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture the webcam
# 0 is default camera
webcam = cv2.VideoCapture(0)

while True:
    # Capture frame from webcam
    # successful_frame_read tells you that capturing the frame was successful
    # frame is the actual video frame
    successful_frame_read, frame = webcam.read()

    # Convert frame to grayscle
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get coordinates of face on the fram
    # face_coordinates is a list that stores the coordinates of each and all faces detected
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

    # Draw rectangle around the face (image, top_left_coord, width_and_height_of_face, colorBGR, thickness)
    # (x, y) is the coordinate of the top left corner of the face coordinate.
    # (w, h) is the width and height of the polygon of the face coordinates,
    for (x, y, w, h) in face_coordinates:
        # For each set of face coordinates (for each face), draw  a rectange on the coordinates
        # First argument frame determines where to draw the rectangle
            # Here we draw rectangle into the video frame
        # Second argument (x, y) is the coordinate of the top left corner of the rectangle
        # Fourth argument (0, 0, 255) is the BGR (blue, green, red) value of the rectangle
        # Last argument 2 is the thickness of the rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Show frame
    cv2.imshow('Face Detector', frame)

    # Wait one milisecond for user to hit a key. If user doesn't hit any key, continue to next frame.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the webcam object
webcam.release()

# Destroy all the OpenCV windows
cv2.destroyAllWindows()
