# Reconhecimento Facial OpenCV
Projeto de reconhecimento facial com OpenCV - após treinamento das faces pelo algoritmo será capaz de fazer a detecção por frame

## Começando
Primeiro será necessário realizar a detecção da face e capturar as fotos para o treinamento executando o "dectect_face.py"
Depois será feito o treinamento das face capturadas com o "treinamento.py"
E no final usamos algoritmo Eigenface para fazer o reconhecimento

### Pré-requisitos
Instale todas as dependências do requirements.txt com PIP


### Instalação
Crie um virtual env para empacotar suas libs Python

```
python -m venv venv
```

Ative sua virtual env

```
source venv/bin/active
```

Use o PIP, gerenciador de pacotes do Python para instalar todos os requisitos

```
pip install -r requirements.txt
```

## Uso do programa
1 - Primeiro faça as capturas salvando as imagens das faces detectadas:

```
python detect_face.py
```
digite um numero para ser o identificador da face e clique na tecla "q" para salvar a imagem da face detectada.


2 - Faça o apredizado das faces detectadas:

```
python treinamento.py
```
3 - Execute o reconhecedor facil

```
python reconhecedor_eigenfaces.py
```

## Built With
* [OpenCV](https://pypi.org/project/opencv-contrib-python/) - OpenCV
* [Numpy](https://numpy.org/) - Numpy
* [Python](https://www.python.org/) - Python
