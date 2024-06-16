import cv2
import time
import sounddevice as sd
import numpy as np

# Carrega o classificador Haar para detecção de olhos
classificadorOlhos = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

# Armazena o tempo da última detecção de olhos
tempoUltimaDeteccaoOlhos = time.time()

# Inicia a captura de vídeo da webcam (dispositivo padrão, geralmente ID 0)
captura = cv2.VideoCapture(0)

# Função para tocar um alarme sonoro
def tocarAlarme(frequencia, duration):
    t = np.linspace(0, duration, int(44100 * duration), endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * frequencia * t)
    sd.play(x, 44100)

# Loop principal para a captura e processamento de frames de vídeo
while True:
    # Captura um frame da webcam
    retorno, frame = captura.read()
    if not retorno:
        break

    # Converte o frame para escala de cinza
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detecta olhos no frame em escala de cinza
    olhos = classificadorOlhos.detectMultiScale(cinza, 1.3, 5)

    # Se olhos forem detectados, atualiza o tempo da última detecção
    if len(olhos) > 0:
        tempoUltimaDeteccaoOlhos = time.time()
        # Desenha retângulos ao redor dos olhos detectados
        for (x, y, largura, altura) in olhos:
            cv2.rectangle(frame, (x, y), (x+largura, y+altura), (255, 0, 0), 2)

    # Exibe o frame com as detecções
    cv2.imshow('Detecção de Olhos', frame)

    # Se passaram mais de 3 segundos desde a última detecção de olhos, toca o alarme numa frequencia de 500Hz a cada 0.5 segundos (120 BPM)
    if time.time() - tempoUltimaDeteccaoOlhos > 3:
        tocarAlarme(500, 0.5)
        time.sleep(0.5)

    # Verifica se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fecha todas as janelas abertas
captura.release()
cv2.destroyAllWindows()