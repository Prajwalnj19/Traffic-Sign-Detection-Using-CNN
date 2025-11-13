import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Settings
frameWidth = 640
frameHeight = 480
brightness = 180
confidence_threshold = 0.90  # Stronger threshold
font = cv2.FONT_HERSHEY_SIMPLEX

# Load model
model = load_model("model.h5")

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    return img / 255

def getCalssName(classNo):
    classNames = [ 'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons']

    return classNames[classNo] if 0 <= classNo < len(classNames) else "Unknown"

# Setup camera
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

while True:
    success, imgOrignal = cap.read()

    # Optional: Draw a central ROI box
    roi_start = (220, 140)
    roi_end = (420, 340)
    cv2.rectangle(imgOrignal, roi_start, roi_end, (255, 255, 0), 2)
    roi = imgOrignal[roi_start[1]:roi_end[1], roi_start[0]:roi_end[0]]

    # Preprocess
    img = cv2.resize(roi, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    # Predict
    predictions = model.predict(img)[0]
    classIndex = np.argmax(predictions)
    confidence = np.max(predictions)

    if confidence > confidence_threshold:
        result_text = f"CLASS: {classIndex} {getCalssName(classIndex)}"
        prob_text = f"PROBABILITY: {round(confidence * 100, 2)}%"
    else:
        result_text = "❌ No valid traffic sign detected"
        prob_text = f"Confidence: {round(confidence * 100, 2)}%"

    # Display results
    cv2.putText(imgOrignal, result_text, (20, 35), font, 0.75, (0, 0, 255), 2)
    cv2.putText(imgOrignal, prob_text, (20, 75), font, 0.75, (1, 1, 305), 2)
    cv2.imshow("Result", imgOrignal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
