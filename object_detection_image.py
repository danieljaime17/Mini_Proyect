import cv2  # Importamos la librería OpenCV para trabajar con imágenes y videos
import numpy as np  # Importamos NumPy para trabajar con matrices y cálculos numéricos

# Cargamos el archivo prototxt que define la estructura del modelo pre-entrenado
prototxt = "model/MobileNetSSD_deploy.prototxt.txt"

# Cargamos el archivo del modelo pre-entrenado
model = "model/MobileNetSSD_deploy.caffemodel"

# Definimos las clases que el modelo puede detectar
classes = {0: "background", 1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle",
           6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow", 11: "diningtable", 12: "dog",
           13: "horse", 14: "motorbike", 15: "person", 16: "pottedplant", 17: "sheep",
           18: "sofa", 19: "train", 20: "tvmonitor"}

# Cargamos el modelo pre-entrenado usando OpenCV
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Cargamos la imagen desde la ruta especificada
image_path = "imagesVideos/Foto1.jpg"
image = cv2.imread(image_path)  # Leemos la imagen desde el archivo

cv2.imshow("Image", image)  # Mostramos la imagen original en una ventana
cv2.waitKey(0)  # Esperamos a que el usuario presione una tecla para continuar

# Obtenemos las dimensiones de la imagen (alto, ancho y canales de color)
height, width, _ = image.shape

# Redimensionamos la imagen a 300x300 píxeles para que sea compatible con el modelo
image_resized = cv2.resize(image, (300, 300))

# Creamos un "blob" a partir de la imagen redimensionada, que es un formato que el modelo puede entender
blob = cv2.dnn.blobFromImage(image_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5))

print("Blob shape:", blob.shape)  # Mostramos la forma del blob (dimensiones)

# Pasamos el blob como entrada al modelo
net.setInput(blob)

# Realizamos una pasada hacia adelante en la red para obtener las detecciones
detections = net.forward()

# Iteramos sobre cada detección obtenida
for detection in detections[0][0]:
    print(detection)  # Mostramos los datos de la detección

    # Obtenemos el índice de la clase detectada y buscamos su nombre en el diccionario de clases
    label = classes[int(detection[1])]
    print("Label:", label)  # Mostramos el nombre de la clase detectada

    # Calculamos las coordenadas del cuadro delimitador (bounding box) en la imagen original
    box = detection[3:7] * np.array([width, height, width, height])
    (startX, startY, endX, endY) = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    # Dibujamos el cuadro delimitador en la imagen original
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Mostramos la imagen con los objetos detectados en una ventana
cv2.imshow("Detected Objects", image)
cv2.waitKey(0)  # Esperamos a que el usuario presione una tecla para cerrar la ventana
cv2.destroyAllWindows()  # Cerramos todas las ventanas de OpenCV












