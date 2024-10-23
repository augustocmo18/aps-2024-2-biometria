import cv2
from datetime import datetime
import os
import numpy as np

# Caminho haarcascade
detectorFace = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
detectorOlho = cv2.CascadeClassifier('cascade/haarcascade-eye.xml')

# Instanciado Eigen Faces Recognizer
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classifier/classificadorEigen.yml")

def getImageWithId():
    '''
        Percorrer diretorio fotos, ler todas imagens com CV2 e organizar
        conjunto de faces com seus respectivos ids
    '''
    pathsImages = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []
    names = []
    sec_ids = []

    for pathImage in pathsImages:
        imageFace = cv2.cvtColor(cv2.imread(pathImage), cv2.COLOR_BGR2GRAY)
        parts = os.path.split(pathImage)[-1].split('.')
        id = int(parts[1])  # Assuming id is at index 1
        name = parts[2]      # Assuming name is at index 2
        sec_id = int(parts[3])  # Assuming sec_id is at index 3

        ids.append(id)
        names.append(name)
        sec_ids.append(sec_id)
        faces.append(imageFace)

        
        cv2.waitKey(10)
    return np.array(ids), faces, names, np.array(sec_ids)



height, width = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)
ids, faces, names, sec_ids = getImageWithId()

while (True):
    conectado, imagem = camera.read()
    imageGray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Deteccao da face baseado no haarcascade
    faceDetect = detectorFace.detectMultiScale(
        imageGray,
        scaleFactor=1.5,
        minSize=(35, 35),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, h, w) in faceDetect:
        # Desenhando retangulo da face
        cv2.rectangle(imagem, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Detector Olho with face
        region = imagem[y:y+h, x:x+w]
        imageOlhoGray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        olhoDetector = detectorOlho.detectMultiScale(imageOlhoGray)

        for(ox, oy, oh, ow) in olhoDetector:

            # Desenhando retangulo do olho da face detectada
            cv2.rectangle(region, (ox, oy), (ox+ow, oy+oh), (0, 0, 255), 2)
            image = cv2.resize(imageGray[y:y+h, x:x+w], (width, height))

            # Fazendo comparacao da imagem detectada
            id, confianca = reconhecedor.predict(image)

            if id in ids:
                index = np.where(ids == id)[0][0]
                name = names[index]
                sec_id = sec_ids[index]
            else:
                name = 'Nao cadastrado'
                sec_id = 'N/A'

            # Escrevendo texto no frame
            cv2.putText(imagem, name, (x, y + (h + 24)), font, 1, (0, 255, 0))
            cv2.putText(imagem, "Nivel de seguranca: " + str(sec_id), (x, y + (h + 43)), font, 1, (0, 0, 255))
            if str(sec_id) in ('2', '3'):
                if str(sec_id) == '3':
                    cv2.putText(imagem, "Acesso Liberado - Diretor", (x, y + (h + 81)), font, 1, (255, 0, 255))
                else:
                    if str(sec_id) == '2':
                        cv2.putText(imagem, "Acesso Liberado - Funcionario", (x, y + (h + 81)), font, 1, (255, 0, 255))
                    else:
                        cv2.putText(imagem, "Acesso Liberado - Visitante", (x, y + (h + 81)), font, 1, (255, 0, 0))
                        
            cv2.putText(imagem, str(confianca), (x, y + (h + 62)), font, 1, (0, 0, 255))

    # Mostrando frame
    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'): break

camera.release()
cv2.destroyAllWindows()
