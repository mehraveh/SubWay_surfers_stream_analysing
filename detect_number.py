
import cv2
import numpy as np
from keras.models import load_model
# Load a pre-trained digit recognition model (you should replace this with your own model)
model = load_model('/Users/mehravehahmadi/yolov5/data/mnist_v2.h5')

# Open the video capture
cap = cv2.VideoCapture('/Users/mehravehahmadi/yolov5/data/images/oo.mov')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.threshold(frame, 120, 255, cv2.THRESH_BINARY)
    # Preprocess the frame (e.g., resize, convert to grayscale, apply filters)

    # Find contours in the preprocessed frame
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("************")
    #print((contours))
    # Iterate through detected contours
    for contour in contours:
        # Filter contours based on area, aspect ratio, etc.
        x, y, w, h = cv2.boundingRect(contour)
        if 10 < w < 100 and 10 < h < 100:
            # Extract the digit from the bounding box
            digit_roi = frame[y:y+h, x:x+w]
            # Resize the digit to a standard size (e.g., 28x28)
            digit_roi = cv2.resize(digit_roi, (28, 28))
            digit_roi = digit_roi / 255.0  # Normalize pixel values

            # Perform digit recognition using your model (replace this with your recognition code)
            prediction = model.predict(np.array([digit_roi.reshape(28, 28, 1)]))
            digit = np.argmax(prediction)

            # Draw bounding box around the detected digit on the original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Optionally, label the recognized digit
            cv2.putText(frame, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print('dd = ' , digit)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
