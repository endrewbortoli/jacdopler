from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
import os
import base64
import threading

app = Flask(__name__)

# 🔹 Load YOLO model
model = YOLO("coral.pt")  # Update the model path if needed

# 🔹 Set up database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///calibration.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# 🔹 Arena field points
ARENA_POINTS = {
    "1_azul": np.array([[3.330, 4.670], [4.490, 2.666], [5.400, 0], [1.770, 0]], dtype=np.float32),
    "1_vermelho": np.array([[14.230, 3.420], [13.000, 2.666], [13.400, 0], [15.800, 0]], dtype=np.float32),
    "3_azul": np.array([[3.330, 3.360], [4.490, 5.370], [5.400, 8.000], [1.770, 8.000]], dtype=np.float32),
    "3_vermelho": np.array([[14.230, 4.666], [13.000, 5.320], [13.400, 8.000], [15.870, 8.000]], dtype=np.float32)
}

# 🔹 Selected arena side
selected_side = "1_azul"
homographies = {}

# 🔹 Camera setup
cap = cv2.VideoCapture(0)
camera_active = True

# 🔹 Database model
class Calibration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    side = db.Column(db.String(20), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    point1_x = db.Column(db.Float, nullable=False)
    point1_y = db.Column(db.Float, nullable=False)
    point2_x = db.Column(db.Float, nullable=False)
    point2_y = db.Column(db.Float, nullable=False)
    point3_x = db.Column(db.Float, nullable=False)
    point3_y = db.Column(db.Float, nullable=False)
    point4_x = db.Column(db.Float, nullable=False)
    point4_y = db.Column(db.Float, nullable=False)

# 🔹 Create database
with app.app_context():
    db.create_all()

# 🔹 Get pov_points from DB
def get_pov_points_from_db(side):
    calibration = Calibration.query.filter_by(side=side).first()
    if calibration:
        return np.array([
            [calibration.point1_x, calibration.point1_y],
            [calibration.point2_x, calibration.point2_y],
            [calibration.point3_x, calibration.point3_y],
            [calibration.point4_x, calibration.point4_y]
        ], dtype=np.float32)
    return None

# 🔹 Compute homography
def compute_homography(side):
    global homographies
    pov_points = get_pov_points_from_db(side)
    if pov_points is None:
        print(f"⚠️ Error: No calibration data found for {side}")
        return None
    field_points = ARENA_POINTS[side]
    homography_matrix, _ = cv2.findHomography(pov_points, field_points)
    homographies[side] = homography_matrix
    return homography_matrix

# 🔹 Process frames with YOLO + Homography
def process_frame():
    global selected_side, camera_active

    while camera_active:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)
        best_coral_position = None
        best_conf = 0

        for result in results:
            for box in result.boxes:
                conf = box.conf[0]
                if conf > 0.4 and conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f'Coral {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    best_coral_position = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)

        if best_coral_position is not None and selected_side in homographies:
            coral_field_position = cv2.perspectiveTransform(np.array([best_coral_position]), homographies[selected_side])
            print(f"📍 Coral Position ({selected_side}): {coral_field_position[0][0]}")

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# 🔹 API - Save calibration
@app.route('/save_calibration', methods=['POST'])
def save_calibration():
    data = request.json
    side = data['side']
    points = data['points']
    image_data = data['image']

    # Convert base64 image to file
    image_filename = f'static/calibration_{side}.png'
    image_path = os.path.join(os.getcwd(), image_filename)
    with open(image_filename, 'wb') as img_file:
        img_file.write(base64.b64decode(image_data.split(',')[1]))

    # Save to DB
    new_calibration = Calibration(
        side=side,
        image_path=image_filename,
        point1_x=points[0]['x'], point1_y=points[0]['y'],
        point2_x=points[1]['x'], point2_y=points[1]['y'],
        point3_x=points[2]['x'], point3_y=points[2]['y'],
        point4_x=points[3]['x'], point4_y=points[3]['y']
    )

    db.session.add(new_calibration)
    db.session.commit()

    # Compute and store homography matrix
    compute_homography(side)

    return jsonify({"message": "Calibration saved successfully!"})

# 🔹 API - Set arena side
@app.route('/set_side', methods=['POST'])
def set_side():
    global selected_side
    selected_side = request.form['side']
    compute_homography(selected_side)
    return jsonify({"message": f"Arena side set to {selected_side}"}), 200

# 🔹 API - Stop camera
@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    cap.release()
    return jsonify({"message": "Camera stopped successfully"}), 200

# 🔹 Web Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calibration')
def calibration():
    return render_template('calibration.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 🔹 Start Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
