import cv2
import numpy as np
from ultralytics import YOLO

def load_yolo_model(model_path):
    """Carrega o modelo YOLO treinado."""
    return YOLO(model_path)

def compute_homography():
    """Define os pontos de referência e calcula a matriz de homografia."""
    pov_points = np.array([
        [335,  356], [444,  336], [903,  292], [151,  418]
    ], dtype=np.float32)
    field_points = np.array([
        [3.321, 3.289], [3.289, 2.658], [5.139, 0.000], [1.743, 0.000]
    ], dtype=np.float32)

    homography_matrix, _ = cv2.findHomography(pov_points, field_points)
    return homography_matrix

def process_webcam(model, homography_matrix):
    """Processa a captura ao vivo da webcam e aplica a detecção do Coral em tempo real."""
    cap = cv2.VideoCapture(0)  # Captura da webcam

    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        best_coral_position = None
        best_conf = 0

        for result in results:
            for box in result.boxes:
                conf = box.conf[0]
                if conf > 0.4 and conf > best_conf:  # Seleciona o Coral com maior confiança
                    best_conf = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    label = f'Coral {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    best_coral_position = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)

        if best_coral_position is not None:
            best_coral_position = np.array([best_coral_position])
            coral_field_position = cv2.perspectiveTransform(best_coral_position, homography_matrix)
            print(f"Posição estimada do Coral com maior confiança no campo FRC: {coral_field_position}")

        # Exibir a captura ao vivo
        cv2.imshow("Detecção de Coral em Tempo Real", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    model_path = "coral.pt"  # Ajuste o caminho correto

    model = load_yolo_model(model_path)
    homography_matrix = compute_homography()

    process_webcam(model, homography_matrix)

if __name__ == "__main__":
    main()