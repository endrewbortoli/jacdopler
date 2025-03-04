import cv2
import numpy as np

def process_image(image):
    # Converte para HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Cria a máscara com os novos limites
    mask = cv2.inRange(hsv, lower_coral, upper_coral)
    
    # Operações morfológicas
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    # Encontra contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calcula áreas mínima e máxima
    height, width = image.shape[:2]
    image_area = height * width
    min_area = image_area * (cv2.getTrackbarPos("Area Min%", "Ajuste HSV") / 1000)  # 0.1% a 10%
    max_area = image_area * (cv2.getTrackbarPos("Area Max%", "Ajuste HSV") / 1000)
    
    # Desenha os resultados
    result = image.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Filtra por tamanho
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result, "CORAL", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    return result, mask, min_area, max_area

# VALORES INICIAIS PARA BRANCO
lower_coral = np.array([0, 0, 200])
upper_coral = np.array([179, 15, 255])

cv2.namedWindow("Ajuste HSV")
# Trackbars para cores
cv2.createTrackbar("H Min", "Ajuste HSV", 0, 179, lambda x: None)
cv2.createTrackbar("H Max", "Ajuste HSV", 179, 179, lambda x: None)
cv2.createTrackbar("S Min", "Ajuste HSV", 0, 255, lambda x: None)
cv2.createTrackbar("S Max", "Ajuste HSV", 30, 255, lambda x: None)
cv2.createTrackbar("V Min", "Ajuste HSV", 200, 255, lambda x: None)
cv2.createTrackbar("V Max", "Ajuste HSV", 255, 255, lambda x: None)

# Novos trackbars para área (0.1% a 10% da imagem)
cv2.createTrackbar("Area Min%", "Ajuste HSV", 4, 100, lambda x: None)  # 0.1% = 1
cv2.createTrackbar("Area Max%", "Ajuste HSV", 20, 100, lambda x: None)  # 5% = 50

while True:
    image_path = input("Digite o caminho da imagem (ou 'sair' para encerrar): ")
    if image_path.lower() == 'sair':
        break
    
    image = cv2.imread(image_path)
    if image is None:
        print("Erro: Imagem não encontrada ou formato inválido!")
        continue
    
    while True:
        # Atualiza valores
        h_min = cv2.getTrackbarPos("H Min", "Ajuste HSV")
        h_max = cv2.getTrackbarPos("H Max", "Ajuste HSV")
        s_min = cv2.getTrackbarPos("S Min", "Ajuste HSV")
        s_max = cv2.getTrackbarPos("S Max", "Ajuste HSV")
        v_min = cv2.getTrackbarPos("V Min", "Ajuste HSV")
        v_max = cv2.getTrackbarPos("V Max", "Ajuste HSV")
        
        lower_coral = np.array([h_min, s_min, v_min])
        upper_coral = np.array([h_max, s_max, v_max])
        
        # Processa imagem e obtém áreas
        result, mask, min_area, max_area = process_image(image)
        
        # Adiciona informações na tela
        cv2.putText(result, f"Area Min: {int(min_area)}px", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.putText(result, f"Area Max: {int(max_area)}px", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.imshow("Imagem Original", image)
        cv2.imshow("Resultado", result)
        cv2.imshow("Mascara HSV", mask)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            break
        elif key == ord('s'):
            cv2.imwrite("resultado.jpg", result)
            print("Resultado salvo como 'resultado.jpg'")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()

cv2.destroyAllWindows()