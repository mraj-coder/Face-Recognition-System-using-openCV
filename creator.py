import numpy as np
import cv2

igniter = 8880  # just a random number to avoid overwriting file names
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to capture images from the camera
def capture_images(id):
    cap = cv2.VideoCapture(0)  # Use index 0 for the default camera
    if not cap.isOpened():  # Check if the camera is opened successfully
        print("Error: Unable to access the camera.")
        return

    sampleNum = 0
    while True:
        ret, img = cap.read()
        if not ret:  # Check if frame is read correctly
            print("Error: Unable to capture frame.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            sampleNum += 1
            cv2.imwrite(f"dataSet/User.{id}.{sampleNum * igniter}.jpg", gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

        if sampleNum > 210:
            break

    cap.release()
    cv2.destroyAllWindows()

    print('Collection complete!!!')

def main():
    id = input('Enter the ID #: ')
    capture_images(id)

if __name__ == "__main__":
    main()
