import cv2
import numpy as np
import dlib
from math import hypot

#Iniciando a Captura de Video
cap = cv2.VideoCapture(0)

#Carregando a Imagem do Nariz de Palhaço
narizDePalhaco = cv2.imread("./imagem/narizDePalhaco.png")

#Carregando o Detector de Faces
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./predictor/shape_predictor_68_face_landmarks.dat")

_, frame = cap.read()
rows, cols, _ = frame.shape
mascaraNariz = np.zeros((rows, cols), np.uint8)

while True:
    _, frame = cap.read()
    mascaraNariz.fill(0)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Detectando o Rosto com o detector
    faces = detector(frame)
    for face in faces:
        
        #Detectando os Pontos Faciais
        landmarks = predictor(gray_frame, face)
        
        #Pegando as Coordenadas do local do Nariz
        topoNariz = (landmarks.part(29).x, landmarks.part(29).y)
        centroNariz = (landmarks.part(30).x, landmarks.part(30).y)
        esquerdaNariz = (landmarks.part(31).x, landmarks.part(31).y)
        direitaNariz = (landmarks.part(35).x, landmarks.part(35).y)
        
        #Pegando a Largura e altura do Nariz com base nas Coordenadas
        larguraNariz = int(hypot(esquerdaNariz[0] - direitaNariz[0], esquerdaNariz[1] - direitaNariz[1]) * 1.7)
        alturaNariz = int(larguraNariz * 0.77)

        #Calculando a posição de onde será colocada a imagem do nariz
        pontoTopoEsquerda = (int(centroNariz[0] - larguraNariz / 2), int(centroNariz[1] - alturaNariz / 2))
        pontoBaixoDireita = (int(centroNariz[0] + larguraNariz / 2), int(centroNariz[1] + alturaNariz / 2))

        #Inserindo o Nariz na imagem como um filtro
        narizFiltro = cv2.resize(narizDePalhaco, (larguraNariz, alturaNariz))
        narizFiltroCinza = cv2.cvtColor(narizFiltro, cv2.COLOR_BGR2GRAY)

        _, mascaraNariz = cv2.threshold(narizFiltroCinza, 25, 255, cv2.THRESH_BINARY_INV)
        
        #Pegando a area do nariz e colocando a mascara em cima do frame
        areaNariz = frame[pontoTopoEsquerda[1]: pontoTopoEsquerda[1] + alturaNariz, pontoTopoEsquerda[0]: pontoTopoEsquerda[0] + larguraNariz]
        nose_area_no_nose = cv2.bitwise_and(areaNariz, areaNariz, mask=mascaraNariz)
        narizFinal = cv2.add(nose_area_no_nose, narizFiltro)
        frame[pontoTopoEsquerda[1]: pontoTopoEsquerda[1] + alturaNariz, pontoTopoEsquerda[0]: pontoTopoEsquerda[0] + larguraNariz] = narizFinal
    
    #Mostrando a Imagem final
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break               