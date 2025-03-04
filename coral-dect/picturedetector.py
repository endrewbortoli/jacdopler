import cv2
import numpy as np
from ultralytics import YOLO


def load_yolo_model(model_path):
    return YOLO(model_path)

def detect_coral(model, image_path):
    """Detecta o Coral na imagem e retorna sua posição."""
    img = cv2.imread(image_path)
    results = model(img)
    
    best_conf = 0
    coral_position = None
    for result in results:
        for box in result.boxes:
            conf = box.conf[0]
            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                coral_position = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
                
                # Desenhar a detecção
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f'Coral {conf:.2f}'
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return img, coral_position

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

def transform_coral_position(homography_matrix, coral_position):
    """Transforma a posição do Coral para coordenadas do campo FRC."""
    if coral_position is not None:
        coral_position = np.array([coral_position])
        return cv2.perspectiveTransform(coral_position, homography_matrix)
    return None

def main():
    model_path = "coral.pt"  # Ajuste para o caminho correto
    image_path = "pic6.png"  # Imagem de entrada
    
    model = load_yolo_model(model_path)
    homography_matrix = compute_homography()
    
    img, coral_pov_position = detect_coral(model, image_path)
    coral_field_position = transform_coral_position(homography_matrix, coral_pov_position)
    
    # Exibir resultados
    cv2.imshow("Detecção do Coral", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if coral_field_position is not None:
        print(f"Posição estimada do Coral no campo FRC: {coral_field_position}")
    else:
        print("Nenhum Coral detectado corretamente.")

if __name__ == "__main__":
    main()
