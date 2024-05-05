import os

def list_files(directory, output_file):
    # Abrir archivo de texto para escribir los nombres de los archivos
    with open(output_file, 'w') as file:
        # os.walk recorre el directorio y sus subdirectorios
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                # Escribir la ruta completa del archivo en el archivo de texto
                file_path = os.path.join(dirpath, filename)
                file.write(file_path + '\n')

# Llamar a la función con el directorio deseado y el nombre del archivo de texto de salida
# list_files(r'C:\Users\anhernan\Python\DeepLearning\S14\UFC-5', 'file_list.txt')

import cv2 as cv
import numpy as np

def crop_csquare(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=0, resize=(64, 64)):
    '''
    Carga un video desde el disco y extrae un número específico de frames, los cuales pueden ser redimensionados.

    Parámetros:
    - path (str): Ruta completa al archivo de video.
    - max_frames (int): Número máximo de frames a retornar. Si es 0, se retornan todos los frames del video.
    - resize (tuple of int): Dimensiones (ancho, alto) a las cuales cada frame será redimensionado.

    Retorna:
    - numpy.ndarray: Un arreglo de NumPy conteniendo los frames extraídos y procesados del video. Cada frame es un arreglo tridimensional en el formato BGR de OpenCV.
    '''
    cap = cv.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_csquare(frame)
            frame = cv.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

# #Probemos nuestra función y verifiquemos la forma del array (tensor)
# video_path=r"C:\Users\anhernan\Python\DeepLearning\S14\UFC-5\Archery\v_Archery_g01_c01.avi"
# frames=load_video(video_path)
# print(frames.shape)