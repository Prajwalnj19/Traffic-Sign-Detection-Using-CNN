import cv2
import numpy as np
from flask import Flask, render_template, request, Response
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
model = load_model("model.h5")

# Prediction helpers
def grayscale(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def equalize(img): return cv2.equalizeHist(img)
def preprocessing(img): return equalize(grayscale(img)) / 255
def getClassName(classNo):
    classNames = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
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

# Camera generator
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        roi = frame[140:340, 220:420]
        img = cv2.resize(roi, (32, 32))
        img = preprocessing(img)
        img = img.reshape(1, 32, 32, 1)

        predictions = model.predict(img)[0]
        classIndex = np.argmax(predictions)
        confidence = np.max(predictions)

        if confidence > 0.75:
            label = f"{getClassName(classIndex)} ({round(confidence*100, 2)}%)"
        else:
            label = "❌ No valid traffic sign"

        cv2.rectangle(frame, (220, 140), (420, 340), (255, 255, 0), 2)
        cv2.putText(frame, label, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (1, 1, 355), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['file']
    basepath = os.path.dirname(__file__)
    filepath = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    f.save(filepath)

    from tensorflow.keras.preprocessing import image
    img = image.load_img(filepath, target_size=(32, 32))
    img = np.asarray(img)
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    predictions = model.predict(img)[0]
    classIndex = np.argmax(predictions)
    confidence = np.max(predictions)

    if confidence < 0.75:
        return "❌ No valid traffic sign detected"
    return f"{getClassName(classIndex)} ({round(confidence*100, 2)}%)"

if __name__ == "__main__":
    app.run(debug=True, port=5001)
