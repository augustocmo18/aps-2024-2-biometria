import cv2
import os
import numpy as np

# Usando 3 algoritmos de face detect
eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

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

        cv2.imshow("Face", imageFace)
        cv2.waitKey(10)
    return np.array(ids), faces, names, np.array(sec_ids)


ids, faces, names, sec_ids = getImageWithId()

# Gerando classifier do treinamento
print("Treinando....")
eigenface.train(faces, ids)
eigenface.write('classifier/classificadorEigen.yml')

#fisherface.train(faces, ids)
#fisherface.write('classifier/classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classifier/classificadorLBPH.yml')
print('Treinamento concluído com sucesso!')