import cv2
import numpy as np
from networktables import NetworkTables

# Configuração do NetworkTables
NetworkTables.initialize(server='roboRIO-XXXX-frc.local')  # Substitua XXXX pelo seu time
sd = NetworkTables.getTable('SmartDashboard')

# Variáveis globais para homografia
homography_matrix = None
calibration_points = []
field_points = [
    [0.0, 0.0],          # Canto inferior esquerdo
    [16.4592, 0.0],      # Canto inferior direito (dimensões FRC padrão)
    [16.4592, 8.2296],   # Canto superior direito
    [0.0, 8.2296]        # Canto superior esquerdo
]

def load_homography():
    global homography_matrix
    try:
        homography_matrix = np.load('homography.npy')
        print("Homografia carregada com sucesso!")
    except:
        print("Arquivo de homografia não encontrado. Use o modo calibração.")
        homography_matrix = None

def save_homography():
    if len(calibration_points) >= 4:
        src = np.array(calibration_points, dtype=np.float32)
        dst = np.array(field_points, dtype=np.float32)
        h_matrix, _ = cv2.findHomography(src, dst)
        np.save('homography.npy', h_matrix)
        global homography_matrix
        homography_matrix = h_matrix
        print("Homografia salva!")
    else:
        print("Selecione 4 pontos para calibração")

def pixel_to_field(x_pixel, y_pixel):
    if homography_matrix is None:
        return None
    point = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, homography_matrix)
    return transformed[0][0]  # Retorna (X,Y) em metros

def calibration_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(calibration_points) < 4:
            calibration_points.append((x, y))
            print(f"Ponto {len(calibration_points)}: ({x}, {y})")

def process_image(image):
    # Obter parâmetros dos trackbars
    h_min = cv2.getTrackbarPos("H Min", "Ajustes")
    h_max = cv2.getTrackbarPos("H Max", "Ajustes")
    s_min = cv2.getTrackbarPos("S Min", "Ajustes")
    s_max = cv2.getTrackbarPos("S Max", "Ajustes")
    v_min = cv2.getTrackbarPos("V Min", "Ajustes")
    v_max = cv2.getTrackbarPos("V Max", "Ajustes")
    
    # Processamento da imagem
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)
    
    # Operações morfológicas
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    # Detecção de contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtragem por área
    height, width = image.shape[:2]
    img_area = height * width
    min_area = img_area * (cv2.getTrackbarPos("Area Min%", "Ajustes")/1000)
    max_area = img_area * (cv2.getTrackbarPos("Area Max%", "Ajustes")/1000)
    
    # Processar detecções
    result = image.copy()
    positions = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            # Geometria do contorno
            x, y, w, h = cv2.boundingRect(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                
                # Conversão para coordenadas do campo
                field_pos = pixel_to_field(cx, cy) if homography_matrix else None
                
                # Desenhar resultados
                cv2.rectangle(result, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(result, "CORAL", (x, y-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                # if field_pos:
                #     positions.append(field_pos)
                #     cv2.putText(result, f"X: {field_pos[0]:.2f}m", (x, y+h+20),
                #               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
                #     cv2.putText(result, f"Y: {field_pos[1]:.2f}m", (x, y+h+40),
                #               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1)
    
                
                if field_pos:
                    positions.append(field_pos)
                    print(f"Posição detectada: X={field_pos[0]:.2f}m, Y={field_pos[1]:.2f}m")

    # Enviar posições para o robô
    if positions:
        flat_pos = [coord for pos in positions for coord in pos]
        sd.putNumberArray("CoralPositions", flat_pos)
    else:
        sd.putNumberArray("CoralPositions", [])
    
    return result, mask

# Configuração da interface
cv2.namedWindow("Ajustes")
cv2.createTrackbar("H Min", "Ajustes", 0, 179, lambda x: None)
cv2.createTrackbar("H Max", "Ajustes", 179, 179, lambda x: None)
cv2.createTrackbar("S Min", "Ajustes", 0, 255, lambda x: None)
cv2.createTrackbar("S Max", "Ajustes", 30, 255, lambda x: None)
cv2.createTrackbar("V Min", "Ajustes", 200, 255, lambda x: None)
cv2.createTrackbar("V Max", "Ajustes", 255, 255, lambda x: None)
cv2.createTrackbar("Area Min%", "Ajustes", 1, 100, lambda x: None)
cv2.createTrackbar("Area Max%", "Ajustes", 50, 100, lambda x: None)

# Carregar homografia existente
load_homography()

while True:
    cmd = input("\n=== MENU PRINCIPAL ===\n1. Iniciar Detecção\n2. Calibrar Homografia\n3. Sair\nEscolha: ")
    
    if cmd == '3':
        break
        
    elif cmd == '2':
        # Modo calibração
        cal_img_path = input("\n=== MODO CALIBRAÇÃO ===\nCaminho da imagem de calibração: ")
        cal_img = cv2.imread(cal_img_path)
        
        if cal_img is None:
            print("Erro ao carregar imagem!")
            continue
            
        calibration_points.clear()
        cv2.namedWindow("Calibração")
        cv2.setMouseCallback("Calibração", calibration_callback)
        print("\nInstruções:")
        print("1. Clique nos 4 cantos do campo na ordem solicitada")
        print("2. Pressione 'S' para SALVAR e voltar ao menu")
        print("3. Pressione 'Q' para CANCELAR\n")
        
        calibrando = True
        while calibrando:
            display_img = cal_img.copy()
            # Desenha pontos e números
            for i, (x,y) in enumerate(calibration_points):
                cv2.circle(display_img, (x,y), 10, (0,0,255), -1)
                cv2.putText(display_img, str(i+1), (x+15,y-15),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            # Mostra instruções na imagem
            cv2.putText(display_img, "Clique nos 4 cantos do campo", (10,30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(display_img, "Ordem: 1-Inferior Esq, 2-Inferior Dir,", (10,60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(display_img, "3-Superior Dir, 4-Superior Esq", (10,90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(display_img, "Pressione S para salvar | Q para cancelar", (10,120),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.imshow("Calibração", display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if len(calibration_points) == 4:
                    save_homography()
                    load_homography()  # Recarrega a nova homografia
                    calibrando = False
                    print("\nCalibração concluída! Retornando ao menu...")
                else:
                    print("É necessário selecionar exatamente 4 pontos!")
            elif key == ord('q'):
                calibrando = False
                print("Calibração cancelada!")
                
        cv2.destroyAllWindows()
        
    elif cmd == '1':
        # Modo detecção
        if homography_matrix is None:
            print("\nAVISO: Homografia não calibrada! As posições serão em pixels.")
        
        img_path = input("\n=== MODO DETECÇÃO ===\nCaminho da imagem: ")
        img = cv2.imread(img_path)
        
        if img is None:
            print("Erro ao carregar imagem!")
            continue
            
        print("\nInstruções:")
        print("- Ajuste os trackbars para isolar o CORAL")
        print("- Pressione N para nova imagem")
        print("- Pressione Q para voltar ao menu\n")
            
        detectando = True
        while detectando:
            result, mask = process_image(img)
            
            # Exibir informações
            h, w = img.shape[:2]
            cv2.putText(result, f"Resolucao: {w}x{h}", (10,30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            status_homografia = "Ativa" if homography_matrix else "Nao calibrada"
            cv2.putText(result, f"Homografia: {status_homografia}", (10,60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            
            cv2.imshow("Imagem", result)
            cv2.imshow("Mascara", mask)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                detectando = False
                print("\nCarregando nova imagem...")
            elif key == ord('q'):
                detectando = False
                print("Retornando ao menu principal...")
        
        cv2.destroyAllWindows()

cv2.destroyAllWindows()