from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy
import os
import base64
import threading

app = Flask(__name__)

# Carregar o modelo YOLO
model = YOLO("coral.pt")  # Ajuste o caminho do modelo conforme necessário

# Configurar vídeo da webcam
cap = cv2.VideoCapture(0)
camera_active = True  # Variável de controle da câmera

# Matrizes de homografia para os quatro lados
homographies = {
    "1_azul": np.eye(3),
    "1_vermelho": np.eye(3),
    "3_azul": np.eye(3),
    "3_vermelho": np.eye(3)
}

selected_side = "1_azul"

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
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    best_coral_position = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)

        if best_coral_position is not None:
            best_coral_position = np.array([best_coral_position])
            coral_field_position = cv2.perspectiveTransform(best_coral_position, homographies[selected_side])
            print(f"Posição estimada do Coral ({selected_side}): {coral_field_position}")

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# 🔹 Encerrar a câmera corretamente ao sair da página
def release_camera():
    global cap, camera_active
    camera_active = False
    cap.release()
    cv2.destroyAllWindows()

# Configuração do banco de dados
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///calibration.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Modelo do banco de dados
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

# Criar banco de dados
with app.app_context():
    db.create_all()

# Rota para salvar a calibração
@app.route('/save_calibration', methods=['POST'])
def save_calibration():
    data = request.json
    side = data['side']
    points = data['points']
    image_data = data['image']

    # Converter a imagem de base64 para arquivo
    image_filename = f'static/calibration_{side}.png'
    image_path = os.path.join(os.getcwd(), image_filename)

    with open(image_filename, 'wb') as img_file:
        img_file.write(base64.b64decode(image_data.split(',')[1]))

    # Criar uma nova entrada no banco de dados
    new_calibration = Calibration(
        side=side,
        image_path=image_filename,
        point1_x=points[0]['x'],
        point1_y=points[0]['y'],
        point2_x=points[1]['x'],
        point2_y=points[1]['y'],
        point3_x=points[2]['x'],
        point3_y=points[2]['y'],
        point4_x=points[3]['x'],
        point4_y=points[3]['y']
    )

    db.session.add(new_calibration)
    db.session.commit()

    return jsonify({"message": "Calibração salva com sucesso!"})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calibration')
def calibration():
    return render_template('calibration.html')

@app.route('/video_feed')
def video_feed():
    global camera_active
    camera_active = True  # Reativa a câmera se necessário
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_side', methods=['POST'])
def set_side():
    global selected_side
    selected_side = request.form['side']
    return jsonify({"message": f"Lado da arena atualizado para {selected_side}"}), 200

# 🔹 Nova rota para encerrar a câmera ao sair da página
@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    release_camera()
    return jsonify({"message": "Câmera encerrada corretamente"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
