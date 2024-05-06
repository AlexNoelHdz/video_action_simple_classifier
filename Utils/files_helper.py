import os
import tensorflow as tf
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

def list_files(directory, output_file):
    '''
    A partir de un directorio devuelve todas 
    '''
    # Abrir archivo de texto para escribir los nombres de los archivos
    with open(output_file, 'w') as file:
        # os.walk recorre el directorio y sus subdirectorios
        for dirpath, dirnames, filenames in os.walk(directory):
            for filename in filenames:
                # Escribir la ruta completa del archivo en el archivo de texto
                file_path = os.path.join(dirpath, filename)
                file.write(file_path + '\n')

# Llamar a la función con el directorio deseado y el nombre del archivo de texto de salida
# list_files(r'C:\Users\anhernan\Python\DeepLearning\S15\UFC-5', 'file_list.txt')

def crop_csquare(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def find_option(class_names, path):
    '''
    Find a class within a path
    '''
    return next((option for option in class_names if option in path), None)

def load_video(path, max_frames=0, resize=(64, 64), class_names=[]):
    '''
    Carga un video desde el disco y extrae un número específico de frames, los cuales pueden ser redimensionados.

    Parámetros:
    - path (str): Ruta completa al archivo de video.
    - max_frames (int): Número máximo de frames a retornar. Si es 0, se retornan todos los frames del video.
    - resize (tuple of int): Dimensiones (ancho, alto) a las cuales cada frame será redimensionado.

    Retorna:
    - numpy.ndarray: Un arreglo de NumPy conteniendo los frames extraídos y procesados del video. Cada frame es un arreglo tridimensional en el formato BGR de OpenCV.
    '''
    # Extraer el nombre de la clase del path del archivo
    class_name = find_option(class_names, path)
    try:
        label = class_names.index(class_name)
    except ValueError:
        print(f"Class name {class_name} not found in CLASS_NAMES")
        return None, None  # Devolver None para ambos, video y etiqueta, si no se encuentra la clase
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
    if len(frames) < max_frames:
        print(f"Video {path} had only {len(frames)} frames, expected {max_frames}")
        return None, None  # Devolver None si no se alcanza el número de frames esperado

    return np.array(frames), label

def load_data_paths(file_path):
    with open(file_path, 'r') as file:
        paths = [line.strip() for line in file.readlines()]
    return paths

def make_dataset(paths, resize_dim, batch_size, max_frames, autotune, class_names):
    def generator():
        for path in paths:
            frames, label = load_video(path, max_frames,resize_dim, class_names)
            if frames is not None and label is not None:
                yield frames, label
            else:
                print(f"Skipping video {path} due to issues.")

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=((None, *resize_dim, 3), ())
    )
    dataset = dataset.padded_batch(batch_size, padded_shapes=((max_frames, *resize_dim, 3), ())).prefetch(autotune)
    return dataset

# CLASS_INDEX = {0:"Archery", 1:"Basketball", 2:"Diving", 3:"PlayingCello", 4:"VolleyballSpiking"}
# CLASS_NAMES = ["Archery", "Basketball", "Diving", "PlayingCello", "VolleyballSpiking"]
# TEST_SIZE = 0.3
# VAL_SIZE = 0.2  # From train how much to validation
# FILE_LIST_PATH = r'S15/file_list.txt'
# RESIZE_DIM = (64, 64)
# BATCH_SIZE = 9
# MAX_FRAMES = 20
# AUTOTUNE = tf.data.AUTOTUNE

# # Carga de rutas de los videos
# video_paths = load_data_paths(FILE_LIST_PATH)

# # Dividir los datos en entrenamiento, validación y pruebas
# train_paths, test_paths = train_test_split(video_paths, test_size=TEST_SIZE, random_state=42)
# train_paths, val_paths = train_test_split(train_paths, test_size=VAL_SIZE, random_state=42)

# train_dataset = make_dataset(train_paths, RESIZE_DIM, BATCH_SIZE, MAX_FRAMES, AUTOTUNE, CLASS_NAMES)
# val_dataset = make_dataset(val_paths, RESIZE_DIM, BATCH_SIZE, MAX_FRAMES, AUTOTUNE, CLASS_NAMES)
# test_dataset = make_dataset(test_paths, RESIZE_DIM, BATCH_SIZE, MAX_FRAMES, AUTOTUNE, CLASS_NAMES)

# # Recuperar ejemplo
# # Supongamos que train_dataset es tu tf.data.Dataset
# for video_batch, label_batch in train_dataset.take(1):  # Tomamos solo un batch para el ejemplo
#     print("Número de videos en el batch, número de frames, alto, ancho, canales: ")
#     print("Videos batch shape:", video_batch.shape)
#     print("Labels batch:\n")  # Usando.numpy() para visualizar las etiquetas
#     for index in label_batch.numpy():
#         print(f"Clase({index}):{CLASS_INDEX[index]}")