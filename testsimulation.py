print("Setting up...")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import eventlet.wsgi
import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
import base64
from io import BytesIO
from PIL import Image
import cv2

sio = socketio.Server()
app = Flask(__name__)
maxSpeed = 10

def preProcess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (200,66))
    img = img / 255.0
    return img

# @sio.on("telemetry")
# def telemetry(sid, data):
#     speed = float(data['speed'])
#     image = Image.open(BytesIO(base64.b64decode(data['image'])))
#     image = np.asarray(image)
#     image = preProcess(image)
#     image = np.array([image])
#     steering = float(model.predict(image))
#     throttle = 1.0 - speed / maxSpeed
#     print('{} {} {}'.format(steering, throttle, speed))
#     sendControl(steering, throttle)

@sio.on('connect')
def connect(sid, environ):
    print(f'Connected to simulator with session id: {sid}')
    sendControl(0, 0)

@sio.on('telemetry')
def telemetry(sid, data):
    print('Received telemetry data')
    if data:
        speed = float(data['speed'])
        print(f'Current Speed: {speed}')
        steering = 0.0  # Constant steering angle
        throttle = 0.6  # Constant throttle
        print(f'Sending: Steering: {steering:.4f}, Throttle: {throttle:.4f}, Current Speed: {speed:.2f}')
        sendControl(steering, throttle)
    else:
        print('No data received in telemetry')

def sendControl(steering, throttle):
    print(f'Sending control: steering={steering}, throttle={throttle}')
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })

if __name__ == '__main__':
    try:
        print("Loading model...")
        model = load_model('model.h5', custom_objects={'mean_squared_error': MeanSquaredError})
        print("Model loaded successfully.")
        app = socketio.Middleware(sio, app)
        print("Starting server on port 4567...")
        eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
    except Exception as e:
        print(f"An error occurred: {str(e)}")