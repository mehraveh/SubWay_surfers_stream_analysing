import cv2
import numpy as np

# Read the image
image = cv2.imread('output_0087.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection (you can use other methods like Canny)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Use HoughLines to detect lines in the image
lines = cv2.HoughLines(edges, 1, np.pi, threshold=50)

# Draw the lines on the original image
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
     # Calculate the length of the line
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Print the length of the line
    print(f"Line length: {line_length}")

    
# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
