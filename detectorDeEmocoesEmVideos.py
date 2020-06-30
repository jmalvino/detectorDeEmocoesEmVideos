from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import csv
#%tensorflow_version 2.x
import tensorflow

#---------------------------------------------------------

#DECLARACAO DE VARIAVEIS PARA MUDANCA DE PESSOA
nomeDoVideo = "PASTA\NomeDoArquivo.mp4"
nomeFinalDoVideo = 'resultado_video_teste_NomeDoArquivo.avi'
nomeDoCVC = 'Pasta\saidaDoArquivo.csv'

#---------------------------------------------------------

# Carregando o Modelo
from tensorflow.keras.models import load_model
model = load_model("modelo_02_expressoes.h5")

#---------------------------------------------------------

# Caregando o video
arquivo_video = nomeDoVideo
cap = cv2.VideoCapture(arquivo_video)
conectado, video = cap.read()

# mostra as dimensões em px do video
print(video.shape)

#---------------------------------------------------------

# REDIMENSIONANDO O TAMANHO
redimensionar = True 
largura_maxima = 600  
if (redimensionar and video.shape[1]>largura_maxima): 
  proporcao = video.shape[1] / video.shape[0] 
  video_largura = largura_maxima
  video_altura = int(video_largura / proporcao)
else:
  video_largura = video.shape[1]
  video_altura = video.shape[0]
# se redimensionar = False o video de saida permanecerá com os valores da largura e altura os mesmos do vídeo original  

#---------------------------------------------------------

# DEFININDO AS CONFIGURAÇÕES DO VIDEO
# nome do arquivo de vídeo que será salvo
nome_arquivo = nomeFinalDoVideo 

# definição do codec
fourcc = cv2.VideoWriter_fourcc(*'XVID') 
fps = 24
saida_video = cv2.VideoWriter(nome_arquivo, fourcc, fps, (video_largura, video_altura))

#---------------------------------------------------------

# Processamento do Vídeo e Gravação de resultados
from tensorflow.keras.preprocessing.image import img_to_array
haarcascade_faces ='haarcascade_frontalface_default.xml'
fonte_pequena, fonte_media = 0.4, 0.7
fonte = cv2.FONT_HERSHEY_SIMPLEX
expressoes = ["Raiva", "Nojo", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"] 

raiva = 0
nojo = 0
medo = 0
feliz = 0
triste = 0
surpreso = 0
neutro = 0

while (cv2.waitKey(1) < 0):
    conectado, frame = cap.read()    
    if not conectado:
      # caso aconteça de dá um problema em carregar a imagem, o programa é interrompido
        break  
    t = time.time() 
    if redimensionar:
      frame = cv2.resize(frame, (video_largura, video_altura)) 
    face_cascade = cv2.CascadeClassifier(haarcascade_faces)
    # converte pra grayscale
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(cinza,scaleFactor=1.2, minNeighbors=5,minSize=(30,30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:            
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h+10),(255,50,50),2) #  retângulo da face
            roi = cinza[y:y + h, x:x + w]      # extrai ROI
            roi = cv2.resize(roi, (48, 48))    # redimensiona o tamanho  para 48x48px - tamanho das imagens de treinamento
            roi = roi.astype("float") / 255.0  # normaliza
            roi = img_to_array(roi)            # converte para array
            roi = np.expand_dims(roi, axis=0)  # muda o shape da array
            
            # faz a predição - calcula as probabilidades
            result = model.predict(roi)[0]  

            #print(result) 
            if result is not None:
                # encontra a emoção com maior probabilidade
                resultado = np.argmax(result) 
                # escreve a emoção acima do rosto
                cv2.putText(frame,expressoes[resultado],(x,y-10), fonte, fonte_media,(255,255,255),1,cv2.LINE_AA) 
                # escrever no csv
                with open(nomeDoCVC, 'a', newline='') as saida:                
                  escrever = csv.writer(saida)
                  formatacao = expressoes[resultado]                        
                  if(formatacao == "Triste"):
                    triste = triste+1
                  elif (formatacao == "Raiva"):
                    raiva = raiva+1
                  elif (formatacao == "Nojo"):
                    nojo = nojo+1  
                  elif (formatacao == "Medo"):
                    medo = medo+1  
                  elif (formatacao == "Feliz"):
                    feliz = feliz+1  
                  elif (formatacao == "Surpreso"):
                    surpreso = surpreso+1  
                  elif (formatacao == "Neutro"):
                    neutro = neutro+1  

                  print(formatacao)                          
                  escrever.writerow([formatacao]) 

    # tempo processado = tempo atual (time.time()) - tempo inicial (t)    
    cv2.putText(frame, " frame processado em {:.2f} segundos".format(time.time() - t), (20, video_altura-20), fonte, fonte_pequena, (250, 250, 250), 0, lineType=cv2.LINE_AA)
    cv2.imshow('Detector de emocoes', frame) 
     # grava o frame atual
    saida_video.write(frame)

with open(nomeDoCVC, 'a', newline='') as saida:
    escrever = csv.writer(saida)
    formatacao = "Triste=" + str(triste)                            
    escrever.writerow([formatacao]) 

with open(nomeDoCVC, 'a', newline='') as saida:
    escrever = csv.writer(saida)
    formatacao = "Raiva=" + str(raiva)                   
    escrever.writerow([formatacao]) 

with open(nomeDoCVC, 'a', newline='') as saida:
    escrever = csv.writer(saida)
    formatacao = "Nojo=" + str(nojo)                      
    escrever.writerow([formatacao]) 

with open(nomeDoCVC, 'a', newline='') as saida:
    escrever = csv.writer(saida)
    formatacao = "Medo=" + str(medo)                       
    escrever.writerow([formatacao])    

with open(nomeDoCVC, 'a', newline='') as saida:
    escrever = csv.writer(saida)
    formatacao = "Feliz=" + str(feliz)                        
    escrever.writerow([formatacao]) 

with open(nomeDoCVC, 'a', newline='') as saida:
    escrever = csv.writer(saida)
    formatacao = "Surpreso=" + str(surpreso)                       
    escrever.writerow([formatacao]) 
    
with open(nomeDoCVC, 'a', newline='') as saida:
    escrever = csv.writer(saida)
    formatacao = "Neutro=" + str(neutro)                           
    escrever.writerow([formatacao]) 

print("Terminou")
saida_video.release() 
cv2.destroyAllWindows()