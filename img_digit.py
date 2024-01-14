import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained MNIST model
mnist_model = load_model('/Users/mehravehahmadi/yolov5/data/emnist.h5')

# Load the input image
image = cv2.imread('/Users/mehravehahmadi/yolov5/output_0076.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding (adjust threshold values as needed)
_, binary_image = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)
#blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(binary_image, threshold1=100, threshold2=200)  # Adjust thresholds as needed
#cv2.imshow('Digit Detection Result', edges)


# Find contours in the binary image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

print(len(contours))
# Initialize an empty list to store filtered contours
filtered_contours = []
# Define criteria for filtering contours (adjust these values as needed)
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    print(x, y, w, h)
#if 10 < w < 200 and 10 < h < 200 :
    filtered_contours.append(contour)
print(len(filtered_contours))

# Initialize an empty list to store recognized digits
recognized_digits = []

# Loop through filtered contours and recognize digits
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    digit_roi = binary_image[y:y+h, x:x+w]

    # Resize the digit to a standard size (e.g., 28x28 pixels)
    digit_roi = cv2.resize(digit_roi, (28, 28))
    digit_roi = digit_roi / 255.0  # Normalize pixel values (if needed)

    # Perform digit recognition using the pre-trained MNIST model
    digit_roi = digit_roi.reshape(1, 28, 28, 1)  # Reshape for model input
    predicted_class = np.argmax(mnist_model.predict(digit_roi))

    # Append the recognized digit to the list
    recognized_digits.append((predicted_class, (x, y)))

print(recognized_digits)
# Draw bounding boxes and labels on the original image
for digit, (x, y) in recognized_digits:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, str(digit), (x+10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
cv2.imshow('Digit Detection Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
