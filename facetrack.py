import cv2
import numpy as np

def detect_face_shape(face):
    # Calculate the aspect ratio of the face bounding box
    x, y, w, h = face
    aspect_ratio = float(w) / h

    # Define threshold values for circle, square, and oval
    circle_threshold = 0.9
    square_threshold = 1.1
    oval_threshold_low = 0.9
    oval_threshold_high = 1.1

    # Determine the shape based on the aspect ratio
    if aspect_ratio >= circle_threshold and aspect_ratio <= square_threshold:
        return "Circle"
    elif aspect_ratio < oval_threshold_low:
        return "Oval"
    elif aspect_ratio > oval_threshold_high:
        return "Square"
    else:
        return "Undetermined"

def main():
    # Open the default camera (0) or specify a video file path
    capture = cv2.VideoCapture(0)

    while True:
        # Read a frame from the camera
        ret, frame = capture.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use a pre-trained face detection classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around detected faces and classify face shape
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_shape = detect_face_shape((x, y, w, h))
            cv2.putText(frame, f"Shape: {face_shape}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Face Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
