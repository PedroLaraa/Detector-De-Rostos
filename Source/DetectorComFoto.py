# Biblioteca do opencv que instalamos no terminal utilizando o gerenciador de pacotes pip
import cv2
# Aqui carregamos nosso classificador cascata
face_classifier = cv2.CascadeClassifier('Model/haarcascade_frontalface_default.xml')

imagem_original = cv2.imread('Img/original/testeCv2.jpg')
copia_da_imagem_original = cv2.imread('Img/original/testeCv2.jpg')

# Para que o modelo funcione corretamente é necessário mudar a imagem realizando o processamento de imagem alterando ela para escala de cinza
imagem_original_gray = cv2.cvtColor(imagem_original, cv2.COLOR_BGR2GRAY)

# Variável que armazena objeto(s) com as coordenadas da face encontrada na imagem
faces = face_classifier.detectMultiScale(imagem_original_gray, 1.0485258, 6)

if len(faces) == 0:
    print("Nenhuma face encontrada!")
else:
    for (x,y,w,h) in faces:
        # cv2.rectangle comando utilizado para inserir um retangulo nas coordenadas encontradas pelo modelo cascata
        face_detected = cv2.rectangle(imagem_original, (x,y), (x+w,y+h), (127,0,255), 2)

    # Salva a imagem modificada com a face/rosto encontrado na imagem na pasta modificada    
    cv2.imwrite('Img/modificada/imagem1-modificada-com-face.jpg', face_detected)

    # Comando utilizado para exibir imagem
    cv2.imshow('Face Detectada', face_detected)
    cv2.imshow("Imagem Original",copia_da_imagem_original)
    # Comando utilizado para permitir que as imagems apareçam e fiquem esperando na tela até que o usuário digite o comando "Esc" para fechar as imagens depois que forem visualizadas com calma
    cv2.waitKey(0)
    # Comando utilizado para destruir as imagens visualizadas e não manter elas na memória
    cv2.destroyAllWindows()