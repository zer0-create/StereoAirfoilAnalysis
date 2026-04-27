import os
from ultralytics import YOLO

# Cargar el modelo con los nuevos pesos entrenados
model = YOLO("runs/segment/train9/weights/best.pt")  # Cambia la ruta según el archivo de pesos guardado

# Entrenar el modelo nuevamente con tus datos etiquetados
results = model.train(
    data="data.yaml",  # Ruta al archivo data.yaml
    epochs=50,         # Número de épocas adicionales
    imgsz=640          # Tamaño de las imágenes
)

# Validar el modelo después del entrenamiento
metrics = model.val()

# Realizar predicciones en todas las imágenes de la carpeta de prueba
test_images_dir = "test/images"
for image_name in os.listdir(test_images_dir):
    image_path = os.path.join(test_images_dir, image_name)
    if image_path.endswith(('.jpg', '.png', '.jpeg')):  # Filtrar solo imágenes
        print(f"Procesando: {image_path}")
        results = model(image_path)  # Realizar predicción en la imagen

        # Guardar los resultados de la predicción
        for result in results:
            result.plot(save=True, filename=f"resultados_prediccion/{os.path.basename(image_path)}")  # Guarda las imágenes con detecciones

        # Acceder a los resultados de la predicción
        for result in results:
            xy = result.masks.xy  # Máscara en formato de polígono
            xyn = result.masks.xyn  # Máscara normalizada
            masks = result.masks.data  # Máscara en formato de matriz (num_objects x H x W)

print("Entrenamiento y predicciones completados.")