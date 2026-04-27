#!/usr/bin/env python3
"""
FLUJO AERODINÁMICO COMPLETO AUTOMATIZADO - TFG UNIFICADO
========================================================
Sistema unificado que elimina completamente la intervención manual en el análisis de perfiles:

🔄 FLUJO AUTOMÁTICO COMPLETO:
1) 📐 Medición AoA con cámara ZED e IMU
2) 🔍 Detección de contornos con YOLO entrenado
3) 🎯 Identificación automática del perfil NACA más similar
4) 📊 Generación automática de coordenadas del perfil NACA
5) ⚡ Análisis XFOIL automático completo
6) 🌊 Predicción de campos fluidodinámicos con Deep Learning

✨ CARACTERÍSTICAS CLAVE:
- Workflow 100% automatizado sin input manual
- Guarda y reutiliza automáticamente el último AoA medido
- Identifica automáticamente el perfil NACA desde los contornos
- Ejecuta XFOIL automáticamente con el perfil identificado
- Predice campos de presión y velocidad usando el NACA y AoA detectados
- Guarda todos los resultados con nombres consistentes y únicos

🎮 CONTROLES INTERACTIVOS:
- Modo AoA: R=referencia, S=guardar, D=detección, Q=salir
- Modo Detección: S=guardar contornos, A=volver AoA, Q=salir

📁 ARCHIVOS GENERADOS AUTOMÁTICAMENTE:
- aoa_mediciones.txt - Mediciones de ángulo de ataque
- contornos.dat - Contornos detectados por YOLO
- NACA{número}.dat - Coordenadas del perfil identificado
- xfoil_data_{número}.csv - Resultados aerodinámicos completos
- analisis_completo_NACA_{número}.png - Gráficas del análisis XFOIL
- campos_fluidodinamicos_NACA_{número}_AoA_{ángulo}.png - Campos predichos

🚀 FLUJO COMPLETO TFG:
Medición AoA → Detección YOLO → Identificación NACA → Análisis XFOIL → Predicción Campos
Todo automático, sin intervención manual, desde la medición hasta la predicción final.

Autor: Francisco Garcia - Julio 2025
"""

import cv2
import numpy as np
import pyzed.sl as sl
import math
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from ultralytics import YOLO
import torch
import torch.nn.functional as F
import joblib
from model import *

class FlujoAerodinamicoCompleto:
    def __init__(self):
        # Cámara ZED
        self.zed = sl.Camera()
        self.runtime_params = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.sensors_data = sl.SensorsData()
        
        # Variables AoA
        self.reference_angle = 0.0
        self.current_angle = 0.0
        self.is_reference_set = False
        self.saved_angles = []
          # Variables detección
        self.model = None
        self.mode = "aoa"  # "aoa", "detection"
        self.contours_saved = False
        
        # Variables análisis
        self.perfil_identificado = None
        self.naca_identificado = ""
        self.ultimo_aoa_calculado = 0.0  # Para el análisis de campos
        
        # Variables XFOIL
        self.xfoil_results = []
        
    def init_camera(self):
        """Inicializar cámara ZED con IMU"""
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NONE
        
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("Error: No se puede abrir la cámara ZED")
            return False
        
        # Configurar tracking posicional para IMU
        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.enable_imu_fusion = True
        
        status = self.zed.enable_positional_tracking(tracking_params)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Warning: No se pudo habilitar tracking posicional: {status}")
            print("Continuando sin tracking posicional...")
            
        print("Cámara ZED inicializada correctamente")
        return True
    
    def load_yolo_model(self):
        """Cargar modelo YOLO para detección de contornos"""
        try:
            self.model = YOLO("runs/segment/train9/weights/best.pt")
            print("Modelo YOLO cargado correctamente")
            return True
        except Exception as e:
            print(f"Error cargando modelo YOLO: {e}")
            return False   
         
    # ========== FUNCIONES AoA ==========    
    def get_angle(self):
        """Obtener ángulo de rotación desde el IMU con depuración mejorada"""
        try:
            status = self.zed.get_sensors_data(self.sensors_data, sl.TIME_REFERENCE.IMAGE)
            if status == sl.ERROR_CODE.SUCCESS:
                imu_data = self.sensors_data.get_imu_data()
                
                if imu_data.is_available:
                    pose = imu_data.get_pose()
                    orientation = pose.get_orientation()
                    qx, qy, qz, qw = orientation.get()
                    
                    # Calcular los ángulos de Euler desde el quaternion
                    # Roll (rotación en X) para el ángulo de ataque
                    sinr_cosp = 2.0 * (qw * qx + qy * qz)
                    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
                    roll_rad = math.atan2(sinr_cosp, cosr_cosp)
                    
                    # Pitch (rotación en Y) - también útil para depuración
                    sinp = 2.0 * (qw * qy - qz * qx)
                    if abs(sinp) >= 1:
                        pitch_rad = math.copysign(math.pi / 2, sinp)  # Usar 90 grados si fuera de rango
                    else:
                        pitch_rad = math.asin(sinp)
                    
                    # Yaw (rotación en Z)
                    siny_cosp = 2.0 * (qw * qz + qx * qy)
                    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                    yaw_rad = math.atan2(siny_cosp, cosy_cosp)
                    
                    # Convertir a grados
                    roll_degrees = math.degrees(roll_rad)
                    pitch_degrees = math.degrees(pitch_rad)
                    yaw_degrees = math.degrees(yaw_rad)
                    
                    # Para AoA, típicamente usamos el pitch (inclinación hacia arriba/abajo)
                    # Si el perfil está horizontal, el pitch representará el ángulo de ataque
                    angle_aoa = pitch_degrees
                    
                    # Normalizar ángulo a rango [-180, 180]
                    while angle_aoa > 180:
                        angle_aoa -= 360
                    while angle_aoa <= -180:
                        angle_aoa += 360
                        
                    return angle_aoa
                    
        except Exception as e:
            print(f"Error al leer IMU: {e}")
        
        return 0.0
    
    def set_reference(self):
        """Establecer ángulo de referencia"""
        self.reference_angle = self.get_angle()
        self.is_reference_set = True
        print(f"Referencia AoA establecida: {self.reference_angle:.2f}°")
    
    def save_angle(self):
        """Guardar medición de ángulo"""
        if self.is_reference_set:
            timestamp = datetime.now().strftime("%H:%M:%S")
            total_angle = self.current_angle  # Sin inversión
            
            # Normalizar ángulo total
            while total_angle > 180:
                total_angle -= 360
            while total_angle <= -180:
                total_angle += 360
            
            self.saved_angles.append((timestamp, total_angle))
            self.ultimo_aoa_calculado = total_angle  # Guardar último AoA para análisis de campos
            print(f"AoA guardado {-total_angle:.2f}°")
            
            # Guardar en archivo
            with open("aoa_mediciones.txt", "a") as f:
                f.write(f"{timestamp}: Total={total_angle:.2f}°\n")
        else:
            print("Establece primero la referencia con R!")
    
    def draw_aoa_overlay(self, image):
        """Dibujar overlay para medición AoA"""
        h, w = image.shape[:2]
        center_x, center_y = w//2, h//2
        
        # Ejes cartesianos
        cv2.line(image, (0, center_y), (w, center_y), (0, 0, 255), 2)  # X rojo
        cv2.arrowedLine(image, (center_x, center_y), (center_x + 100, center_y), (0, 0, 255), 3)
        cv2.putText(image, "X+ (0°)", (center_x + 110, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.line(image, (center_x, 0), (center_x, h), (255, 0, 0), 2)  # Y azul
        cv2.arrowedLine(image, (center_x, center_y), (center_x, center_y - 100), (255, 0, 0), 3)
        cv2.putText(image, "Y+ (90°)", (center_x + 10, center_y - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.circle(image, (center_x, center_y), 5, (255, 255, 255), -1)
        
        # Línea del ángulo actual
        if self.is_reference_set:
            angle_rad = math.radians(-self.current_angle)  # Invertir para que vector vaya hacia arriba cuando la cámara sube
            end_x = center_x + int(120 * math.cos(angle_rad))
            end_y = center_y - int(120 * math.sin(angle_rad))  # Restar para que sea positivo cuando vaya hacia arriba
            cv2.arrowedLine(image, (center_x, center_y), (end_x, end_y), (0, 255, 0), 4)
        
        # Información
        if self.is_reference_set:
            text = f"AoA: {-self.current_angle:.2f}°"
            color = (0, 255, 0)
        else:
            text = "Presiona R para referencia"
            color = (0, 255, 255)
            
        cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, "MODO: MEDICION AoA", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(image, "R:Ref S:Guardar D:Deteccion Q:Salir", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if len(self.saved_angles) > 0:
            cv2.putText(image, f"Guardados: {len(self.saved_angles)}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ========== FUNCIONES DETECCIÓN ==========
    def detect_contours(self, frame):
        """Detectar contornos usando YOLO"""
        if self.model is None:
            return frame, []
        
        # Convertir a RGB para YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        
        # Preprocesamiento para contornos
        gray_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        edges = cv2.Canny(blurred_frame, 50, 150)
        
        # Predicción YOLO
        results = self.model(frame_rgb)
        annotated_frame = frame_rgb.copy()
        all_contours = []
        
        for result in results:
            if result.masks is not None:
                for box, mask, conf in zip(result.boxes.xyxy, result.masks.data, result.boxes.conf):
                    if conf > 0.8:  # Confianza > 80%
                        # Redimensionar máscara
                        mask_resized = cv2.resize(mask.cpu().numpy().astype(np.uint8), 
                                                 (annotated_frame.shape[1], annotated_frame.shape[0]))
                        
                        # Máscara sombreada
                        colored_mask = np.zeros_like(annotated_frame, dtype=np.uint8)
                        colored_mask[mask_resized == 1] = (0, 255, 0)
                        annotated_frame = cv2.addWeighted(annotated_frame, 1, colored_mask, 0.5, 0)
                        
                        # Detectar contornos
                        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
                        
                        cv2.drawContours(annotated_frame, filtered_contours, -1, (0, 255, 0), 2)
                        all_contours.extend(filtered_contours)
                        
                        # Caja delimitadora
                        margen = 20
                        x1 = max(0, int(box[0]) - margen)
                        y1 = max(0, int(box[1]) - margen)
                        x2 = min(annotated_frame.shape[1], int(box[2]) + margen)
                        y2 = min(annotated_frame.shape[0], int(box[3]) + margen)
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        return annotated_frame, all_contours
    
    def save_contours(self, contours):
        """Guardar contornos en archivo contornos.dat"""
        if len(contours) > 0:
            filename = "contornos.dat"
            
            with open(filename, "w") as file:
                for contour in contours:
                    np.savetxt(file, contour.reshape(-1, 2), fmt="%.2f", header="Nuevo contorno", comments="")
            
            print(f"Contornos guardados en '{filename}' ({len(contours)} contornos)")
            self.contours_saved = True
        else:
            print("No hay contornos para guardar")
    
    def draw_detection_overlay(self, image, contour_count):
        """Overlay para modo detección"""
        h, w = image.shape[:2]
        cv2.putText(image, "MODO: DETECCION CONTORNOS", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(image, f"Contornos detectados: {contour_count}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, "S:Guardar A:AoA Q:Salir", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # ========== FUNCIONES ANÁLISIS CONTORNOS ==========
    def leer_contornos(self, archivo):
        """Lee los contornos desde un archivo .dat"""
        contornos = []
        with open(archivo, "r") as file:
            contorno_actual = []
            for linea in file:
                if "Nuevo contorno" in linea:
                    if contorno_actual:
                        contornos.append(np.array(contorno_actual, dtype=np.float32))
                        contorno_actual = []
                else:
                    punto = list(map(float, linea.strip().split()))
                    contorno_actual.append(punto)
            if contorno_actual:
                contornos.append(np.array(contorno_actual, dtype=np.float32))
        return contornos

    def calcular_caracteristicas(self, contorno):
        """Calcula características del perfil detectado"""
        # Ordenar puntos por coordenada x
        contorno = contorno[np.argsort(contorno[:, 0])]
        
        # Obtener cuerda
        x_min, x_max = contorno[:, 0].min(), contorno[:, 0].max()
        cuerda = x_max - x_min
        
        # Dividir en parte superior e inferior
        x_coords = np.unique(contorno[:, 0])
        parte_superior = []
        parte_inferior = []
        
        for x in x_coords:
            puntos_en_x = contorno[contorno[:, 0] == x]
            y_max = puntos_en_x[:, 1].max()
            y_min = puntos_en_x[:, 1].min()
            parte_superior.append([x, y_max])
            parte_inferior.append([x, y_min])
        
        parte_superior = np.array(parte_superior)
        parte_inferior = np.array(parte_inferior)
        
        # Calcular espesor
        espesores = parte_superior[:, 1] - parte_inferior[:, 1]
        espesor_maximo = espesores.max()
        posicion_espesor_maximo = parte_superior[np.argmax(espesores), 0]
        
        # Porcentajes
        espesor_maximo_porcentaje = (espesor_maximo / cuerda) * 100
        posicion_espesor_maximo_porcentaje = ((posicion_espesor_maximo - x_min) / cuerda) * 100
        
        # Curvatura media
        curvatura_total = 0
        num_puntos = len(parte_superior) - 2
        for i in range(1, len(parte_superior) - 1):
            p1 = parte_superior[i - 1]
            p2 = parte_superior[i]
            p3 = parte_superior[i + 1]
            
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)
            if a * b * c != 0:
                s = (a + b + c) / 2
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                radio = (a * b * c) / (4 * area)
                curvatura = 1 / radio
                curvatura_total += curvatura
        
        curvatura_media = curvatura_total / num_puntos if num_puntos > 0 else 0
        return espesor_maximo_porcentaje, posicion_espesor_maximo_porcentaje, curvatura_media

    def leer_base_datos(self, archivo):
        """Lee la base de datos de perfiles NACA"""
        perfiles = []
        with open(archivo, "r") as file:
            for i, linea in enumerate(file):
                if i == 0:  # Ignorar encabezado
                    continue
                if linea.strip() and not linea.startswith("#"):
                    datos = linea.strip().split()
                    perfil = {
                        "nombre": datos[0],
                        "espesor_max": float(datos[1]),
                        "pos_espesor_max": float(datos[2]),
                        "curvatura_media": float(datos[3])
                    }
                    perfiles.append(perfil)
        return perfiles

    def encontrar_perfil_mas_parecido(self, perfiles, espesor_max, pos_espesor_max, curvatura_media):
        """Encuentra el perfil NACA más parecido"""
        menor_diferencia = float("inf")
        perfil_mas_parecido = None
        
        for perfil in perfiles:
            diferencia = (
                abs(perfil["espesor_max"] - espesor_max) +
                abs(perfil["pos_espesor_max"] - pos_espesor_max) +
                abs(perfil["curvatura_media"] - curvatura_media)
            )
            if diferencia < menor_diferencia:
                menor_diferencia = diferencia
                perfil_mas_parecido = perfil
        
        return perfil_mas_parecido

    def mostrar_imagen_perfil(self, nombre_perfil, carpeta_imagenes):
        """Muestra la imagen del perfil identificado"""
        ruta_imagen = os.path.join(carpeta_imagenes, f"{nombre_perfil}.png")
        
        if os.path.exists(ruta_imagen):
            imagen = cv2.imread(ruta_imagen)
            window_name = f"Perfil Identificado: {nombre_perfil}"
            cv2.imshow(window_name, imagen)
            print(f"Mostrando imagen del perfil: {nombre_perfil}")
            print("Presiona cualquier tecla para continuar...")
            cv2.waitKey(0)
            cv2.destroyWindow(window_name) 
        else:
            print(f"No se encontró la imagen para el perfil: {nombre_perfil}")

    def ejecutar_analisis_contornos(self):
        """Ejecuta análisis automático de contornos y retorna el perfil identificado"""
        archivo_contornos = "contornos.dat"
        archivo_base_datos = "base_datos/airfoil_properties.dat"
        carpeta_imagenes = "base_datos/airfoil_images"
        
        if not os.path.exists(archivo_contornos):
            print("No se encontró el archivo contornos.dat")
            return None
            
        if not os.path.exists(archivo_base_datos):
            print("No se encontró la base de datos de perfiles")
            return None
        
        print("\n" + "="*60)
        print("ANÁLISIS AUTOMÁTICO DE CONTORNOS")
        print("="*60)
        
        try:
            # Leer contornos
            contornos = self.leer_contornos(archivo_contornos)
            
            if contornos:
                # Analizar primer contorno
                espesor_maximo, posicion_espesor_maximo, curvatura_media = self.calcular_caracteristicas(contornos[0])
                
                # Leer base de datos
                base_datos = self.leer_base_datos(archivo_base_datos)
                
                # Encontrar perfil más parecido
                perfil_mas_parecido = self.encontrar_perfil_mas_parecido(base_datos, espesor_maximo, posicion_espesor_maximo, curvatura_media)
                
                # Mostrar resultados
                print(f"Características del perfil detectado:")
                print(f"  • Espesor máximo: {espesor_maximo:.2f}% de la cuerda")
                print(f"  • Posición del espesor máximo: {posicion_espesor_maximo:.2f}% de la longitud total")
                print(f"  • Curvatura media: {curvatura_media:.4f}")
                
                if perfil_mas_parecido:
                    print(f"\n PERFIL NACA IDENTIFICADO:")
                    print(f"  • Nombre: {perfil_mas_parecido['nombre']}")
                    print(f"  • Espesor máximo: {perfil_mas_parecido['espesor_max']:.2f}%")
                    print(f"  • Posición del espesor máximo: {perfil_mas_parecido['pos_espesor_max']:.2f}%")
                    print(f"  • Curvatura media: {perfil_mas_parecido['curvatura_media']:.4f}")
                    
                    # Mostrar imagen
                    self.mostrar_imagen_perfil(perfil_mas_parecido["nombre"], carpeta_imagenes)
                    
                    # Extraer número NACA del nombre (ej: "NACA4-2412" -> "2412")
                    nombre = perfil_mas_parecido["nombre"]
                    if "NACA4-" in nombre:
                        naca_number = nombre.replace("NACA4-", "")
                        self.naca_identificado = naca_number
                        self.perfil_identificado = perfil_mas_parecido
                        print(f"Número NACA extraído para análisis automático: {naca_number}")
                        return naca_number
                    else:
                        print(f"No se pudo extraer número NACA de: {nombre}")
                        return None
                else:
                    print("No se encontró un perfil parecido en la base de datos.")
                    return None
            else:
                print("No se encontraron contornos válidos en el archivo.")
                return None
                
        except Exception as e:
            print(f"Error durante el análisis: {e}")
            return None

    # ========== FUNCIONES NACA/XFOIL ==========
    def naca4_coordinates(self, naca, num_points=100):
        """Genera coordenadas de un perfil NACA 4 dígitos"""
        m = int(naca[0]) / 100.0  # Máximo coeficiente de curvatura
        p = int(naca[1]) / 10.0   # Posición del máximo coeficiente de curvatura
        t = int(naca[2:]) / 100.0 # Espesor relativo máximo

        x = np.linspace(0, 1, num_points)
        yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

        yc = np.where(x < p, 
                      m / p**2 * (2 * p * x - x**2), 
                      m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc_dx = np.where(x < p, 
                          2 * m / p**2 * (p - x), 
                          2 * m / (1 - p)**2 * (p - x))
        theta = np.arctan(dyc_dx)

        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Combinar coordenadas
        x_coords = np.concatenate([xu[::-1], xl[1:]])
        y_coords = np.concatenate([yu[::-1], yl[1:]])

        return x_coords, y_coords

    def generar_dat(self, naca, num_points=100, filename=None):
        """Genera archivo .dat para perfil NACA"""
        if filename is None:
            filename = f"NACA{naca}.dat"
            
        x, y = self.naca4_coordinates(naca, num_points)

        with open(filename, "w") as f:
            f.write(f"NACA {naca}\n")
            for xi, yi in zip(x, y):
                f.write(f"{xi:.6f} {yi:.6f}\n")

        print(f"Archivo NACA '{filename}' generado con éxito.")
        return filename

    def run_xfoil(self, airfoil, alpha, reynolds, mach):
        """Ejecuta XFOIL para un ángulo específico"""
        input_file = "xfoil_input.txt"
        with open(input_file, "w") as f:
            f.write(f"LOAD {airfoil}\n")
            f.write("PANE\n")
            f.write("PPAR\n")
            f.write("N 160\n")
            f.write("\n\n")
            f.write("OPER\n")
            f.write(f"VISC {reynolds}\n")
            f.write(f"MACH {mach}\n")
            f.write("ITER 200\n")
            f.write("PACC\n")
            f.write("polar_output.txt\n\n")
            f.write(f"ALFA {alpha}\n")
            f.write("\nQUIT\n")

        try:
            subprocess.run(["xfoil.exe"], stdin=open(input_file, "r"), text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error ejecutando XFOIL: {e}")
            return None, None
        except FileNotFoundError:
            print("Error: No se encontró xfoil.exe")
            return None, None

        # Leer resultados
        try:
            with open("polar_output.txt", "r") as f:
                lines = f.readlines()
                if len(lines) > 12:
                    data = lines[-1].split()
                    cl = float(data[1])
                    cd = float(data[2])
                    return cl, cd
        except (FileNotFoundError, IndexError, ValueError) as e:
            print(f"Error leyendo datos XFOIL: {e}")
        return None, None

    def ejecutar_analisis_xfoil_automatico(self, naca_number):
        """Ejecuta análisis XFOIL automático para el NACA identificado"""
        print("\n" + "="*60)
        print("INICIANDO ANÁLISIS XFOIL AUTOMÁTICO")
        print("="*60)
        
        # Verificar que existe xfoil.exe
        if not os.path.exists("xfoil.exe"):
            print("Error: No se encontró xfoil.exe en el directorio actual")
            print("Descarga XFOIL desde: https://web.mit.edu/drela/Public/web/xfoil/")
            return False
        
        # Generar archivo DAT automáticamente
        dat_filename = self.generar_dat(naca_number)
        
        # Configuración automática optimizada
        alpha_min, alpha_max, alpha_step = -5, 15, 1
        reynolds, mach = 2000000, 0.09
        
        print(f"Perfil: NACA {naca_number}")
        print(f"Rango de ángulos: {alpha_min}° a {alpha_max}° (paso: {alpha_step}°)")
        print(f"Reynolds: {reynolds:,.0f}")
        print(f"Mach: {mach}")
        
        # Generar rango de ángulos
        alphas = np.arange(alpha_min, alpha_max + alpha_step, alpha_step)
        total_alphas = len(alphas)
        
        print(f"\n Ejecutando análisis para {total_alphas} ángulos...")
        print("-" * 60)

        # Procesar cada ángulo
        self.xfoil_results = []
        failed_count = 0
        
        for i, alpha in enumerate(alphas):
            print(f"Procesando α = {alpha:6.2f}° [{i+1}/{total_alphas}]", end=" ... ")
            
            cl, cd = self.run_xfoil(dat_filename, alpha, reynolds, mach)
            if cl is not None and cd is not None:
                self.xfoil_results.append({
                    "alpha": alpha, 
                    "reynolds": reynolds, 
                    "mach": mach, 
                    "cl": cl, 
                    "cd": cd
                })
                print(f" Cl={cl:.4f}, Cd={cd:.4f}")
            else:
                failed_count += 1
                print("Falló")

        print("-" * 60)
        print(f"Análisis XFOIL completado: {len(self.xfoil_results)} éxitos, {failed_count} fallos")
        
        if self.xfoil_results:
            self.guardar_resultados_xfoil(naca_number)
            self.mostrar_resumen_xfoil(naca_number)
            return True
        else:
            print("No se obtuvieron resultados válidos de XFOIL")
            return False

    def guardar_resultados_xfoil(self, naca_number):
        """Guarda resultados XFOIL y genera gráficas"""
        if not self.xfoil_results:
            return

        # Crear DataFrame
        df = pd.DataFrame(self.xfoil_results)
          # Guardar CSV
        csv_filename = f"xfoil_data_{naca_number}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Datos XFOIL guardados en: {csv_filename}")
        
        # Generar gráficas
        self.generar_graficas_xfoil(df, naca_number)

    def generar_graficas_xfoil(self, df, naca_number):
        """Genera gráficas de resultados XFOIL"""
        fig = plt.figure(figsize=(18, 12))
        
        # Crear 5 subplots en un grid 2x3, dejando la última posición vacía
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2) 
        ax3 = plt.subplot(2, 3, 3)
        ax4 = plt.subplot(2, 3, 4)
        ax5 = plt.subplot(2, 3, 5)
        
        # Gráfica 1: Cl vs Alpha
        ax1.plot(df['alpha'], df['cl'], 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Ángulo de Ataque (°)')
        ax1.set_ylabel('Coeficiente de Sustentación (Cl)')
        ax1.set_title(f'Cl vs α - NACA {naca_number}')
        ax1.grid(True, alpha=0.3)
        
        # Gráfica 2: Cd vs Alpha
        ax2.plot(df['alpha'], df['cd'], 'r-o', linewidth=2, markersize=4)
        ax2.set_xlabel('Ángulo de Ataque (°)')
        ax2.set_ylabel('Coeficiente de Arrastre (Cd)')
        ax2.set_title(f'Cd vs α - NACA {naca_number}')
        ax2.grid(True, alpha=0.3)
        
        # Gráfica 3: Polar (Cl vs Cd)
        ax3.plot(df['cd'], df['cl'], 'g-o', linewidth=2, markersize=4)
        ax3.set_xlabel('Coeficiente de Arrastre (Cd)')
        ax3.set_ylabel('Coeficiente de Sustentación (Cl)')
        ax3.set_title(f'Polar - NACA {naca_number}')
        ax3.grid(True, alpha=0.3)
        
        # Gráfica 4: Eficiencia (Cl/Cd)
        efficiency = df['cl'] / df['cd']
        ax4.plot(df['alpha'], efficiency, 'm-o', linewidth=2, markersize=4)
        ax4.set_xlabel('Ángulo de Ataque (°)')
        ax4.set_ylabel('Eficiencia (Cl/Cd)')
        ax4.set_title(f'Eficiencia - NACA {naca_number}')
        ax4.grid(True, alpha=0.3)
        
        # Gráfica 5: Cl y Cd juntos
        ax5_twin = ax5.twinx()
        l1 = ax5.plot(df['alpha'], df['cl'], 'b-o', linewidth=2, markersize=4, label='Cl')
        l2 = ax5_twin.plot(df['alpha'], df['cd'], 'r-s', linewidth=2, markersize=4, label='Cd')
        ax5.set_xlabel('Ángulo de Ataque (°)')
        ax5.set_ylabel('Cl', color='b')
        ax5_twin.set_ylabel('Cd', color='r')
        ax5.set_title(f'Cl y Cd - NACA {naca_number}')
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfica
        plot_filename = f"analisis_completo_NACA_{naca_number}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Gráficas guardadas en: {plot_filename}")
        
        # Mostrar gráfica
        plt.show()

    def mostrar_resumen_xfoil(self, naca_number):
        """Muestra resumen de resultados XFOIL"""
        if not self.xfoil_results:
            return
            
        df = pd.DataFrame(self.xfoil_results)
        
        print(f"\n RESUMEN ANÁLISIS XFOIL - NACA {naca_number}")
        print("="*60)
        print(f"Ángulos analizados: {len(df)} puntos")
        print(f"Rango de α: {df['alpha'].min():.1f}° a {df['alpha'].max():.1f}°")
        print(f"Cl máximo: {df['cl'].max():.4f} @ α = {df.loc[df['cl'].idxmax(), 'alpha']:.1f}°")
        print(f"Cl mínimo: {df['cl'].min():.4f} @ α = {df.loc[df['cl'].idxmin(), 'alpha']:.1f}°")
        print(f"Cd mínimo: {df['cd'].min():.4f} @ α = {df.loc[df['cd'].idxmin(), 'alpha']:.1f}°")
        
        # Eficiencia máxima
        efficiency = df['cl'] / df['cd']
        max_eff_idx = efficiency.idxmax()
        print(f"Eficiencia máxima: {efficiency.max():.2f} @ α = {df.loc[max_eff_idx, 'alpha']:.1f}°")
        
        # α para Cl = 0
        cl_zero = df.iloc[(df['cl'] - 0).abs().argsort()[:1]]
        if not cl_zero.empty:
            print(f"α para Cl≈0: {cl_zero['alpha'].iloc[0]:.2f}°")
        
        print("="*60)

    def limpiar_archivos_temporales(self):
        """Elimina archivos temporales de XFOIL"""
        temp_files = ["xfoil_input.txt", "polar_output.txt"]
        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)

    # ========== FUNCIONES ANÁLISIS CAMPOS FLUIDODINÁMICOS ==========
    def extraer_parametros_naca(self, naca_str):
        """
        Extrae los parámetros p y t de un perfil NACA de 4 o 5 dígitos
        
        NACA 4 dígitos (ej: 2412):
        - Dígito 1: m*100 (curvatura máxima)
        - Dígito 2: p*10 (posición de curvatura máxima)
        - Dígitos 3-4: t*100 (espesor máximo)
        
        NACA 5 dígitos (ej: 50410):
        - Dígitos 1-2: Designación de curvatura
        - Dígito 3: p*20 (posición de curvatura máxima)
        - Dígitos 4-5: t*100 (espesor máximo)
        """
        naca_str = naca_str.strip()
        
        if len(naca_str) == 4:
            # NACA 4 dígitos
            print(f"Procesando NACA 4 dígitos: {naca_str}")
            
            # Extraer dígitos
            d1 = int(naca_str[0])  # m
            d2 = int(naca_str[1])  # p
            d34 = int(naca_str[2:4])  # t
            
            # Calcular parámetros
            m = d1 / 100.0  # Curvatura máxima
            p = d2 / 10.0   # Posición de curvatura máxima
            t = d34 / 100.0 # Espesor máximo
            
            print(f"   • m (curvatura): {m:.3f}")
            print(f"   • p (posición): {p:.3f}")
            print(f"   • t (espesor): {t:.3f}")
            
            return p, t
            
        elif len(naca_str) == 5:
            # NACA 5 dígitos
            print(f"Procesando NACA 5 dígitos: {naca_str}")
            
            # Extraer dígitos
            d12 = int(naca_str[0:2])  # Designación de curvatura
            d3 = int(naca_str[2])     # p
            d45 = int(naca_str[3:5])  # t
            
            # Calcular parámetros
            p = d3 / 20.0   # Posición de curvatura máxima (dividido por 20 en NACA 5)
            t = d45 / 100.0 # Espesor máximo
            
            print(f"   • Designación: {d12}")
            print(f"   • p (posición): {p:.3f}")
            print(f"   • t (espesor): {t:.3f}")
            
            return p, t
            
        else:
            raise ValueError(f"Número NACA inválido: {naca_str}. Debe tener 4 o 5 dígitos.")

    def aplicar_filtro_media(self, img_tensor, kernel_size=3):
        """Aplica filtro de media usando convolución en PyTorch"""
        # Crear kernel de media
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        
        # Aplicar convolución con padding para mantener dimensiones
        filtered = F.conv2d(img_tensor.unsqueeze(0).unsqueeze(0), 
                           kernel, 
                           padding=kernel_size//2)
        return filtered.squeeze()

    def ejecutar_analisis_campos_fluidodinamicos(self):
        """Ejecuta análisis de campos fluidodinámicos con el NACA y AoA identificados"""
        if not self.naca_identificado:
            print("No hay perfil NACA identificado para análisis de campos")
            return False
            
        if not os.path.exists("modelo_2d.pth"):
            print("No se encontró el modelo de campos fluidodinámicos: modelo_2d.pth")
            return False
            
        if not os.path.exists("scaler_x.save") or not os.path.exists("scaler_y_list.save"):
            print("No se encontraron los escaladores: scaler_x.save, scaler_y_list.save")
            return False

        print("\n" + "="*60)
        print("INICIANDO ANÁLISIS DE CAMPOS FLUIDODINÁMICOS")
        print("="*60)
        
        try:
            # Importar el modelo (debe estar en el directorio)
            from model import MixtoParam2Image
            
            # Configuración básica
            canales_nombres = ['pressure', 'velocity']
            canales = 2
            alto, ancho = 512, 512
            
            # Cargar modelo y normalizadores
            print("Cargando modelo de campos fluidodinámicos...")
            scaler_x = joblib.load("scaler_x.save")
            scaler_y_list = joblib.load("scaler_y_list.save")

            model = MixtoParam2Image(in_features=3, out_channels=2, alto=alto, ancho=ancho)
            model.load_state_dict(torch.load("modelo_2d.pth", map_location=torch.device('cpu')))
            model.eval()
            
            # Extraer parámetros del NACA identificado
            p, t = self.extraer_parametros_naca(self.naca_identificado)
            
            # Usar último AoA medido o valor por defecto
            aoa = -self.ultimo_aoa_calculado if self.ultimo_aoa_calculado != 0.0 else 0.0
            
            print(f"Parámetros configurados automáticamente:")
            print(f"   • NACA identificado: {self.naca_identificado}")
            print(f"   • p (posición): {p:.3f}")
            print(f"   • t (espesor): {t:.3f}")
            print(f"   • AoA medido: {aoa}°")
            print("-"*60)

            # Predicción
            nuevo_input = np.array([[p, t, aoa]])  # [p, t, AoA]
            nuevo_input_norm = scaler_x.transform(nuevo_input)
            nuevo_input_tensor = torch.tensor(nuevo_input_norm, dtype=torch.float32)

            print("Ejecutando predicción de campos fluidodinámicos...")
            with torch.no_grad():
                y_pred_norm = model(nuevo_input_tensor).cpu().numpy()[0]  # (canales, alto, ancho)

            # Postprocesado
            y_pred_processed = []
            for i in range(canales):
                # Desnormalizar
                img_desnorm = scaler_y_list[i].inverse_transform(
                    y_pred_norm[i].reshape(1, -1)).reshape(alto, ancho)
                y_pred_processed.append(img_desnorm)

            # Visualización
            print("Generando visualización de campos...")
            global_min_pressure = -739.3561
            global_max_pressure = 539.6364
            global_min_velocity = 0
            global_max_velocity = 43.80

            fig, axs = plt.subplots(1, canales, figsize=(15, 6))

            for i in range(canales):
                im = axs[i].imshow(y_pred_processed[i], cmap='viridis', interpolation="lanczos")
                axs[i].set_title(f'Predicción {canales_nombres[i]} - NACA {self.naca_identificado} @ {aoa}°')
                
                # Barra con etiquetas globales
                img_min, img_max = y_pred_processed[i].min(), y_pred_processed[i].max()
                if i == 0:
                    labels = np.linspace(global_min_pressure, global_max_pressure, 5)
                else:
                    labels = np.linspace(global_min_velocity, global_max_velocity, 5)
                ticks = np.linspace(img_min, img_max, 5)
                cbar = fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{v:.2f}" for v in labels])
                
            plt.tight_layout()
            
            # Guardar gráfica
            campos_filename = f"campos_fluidodinamicos_NACA_{self.naca_identificado}_AoA_{aoa:.1f}.png"
            plt.savefig(campos_filename, dpi=300, bbox_inches='tight')
            print(f"Campos fluidodinámicos guardados en: {campos_filename}")
            
            plt.show()
            
            print("Análisis de campos fluidodinámicos completado exitosamente")
            return True
            
        except ImportError:
            print("Error: No se pudo importar el modelo. Asegúrate de que model.py esté disponible.")
            return False
        except Exception as e:
            print(f"Error durante el análisis de campos: {e}")
            return False

    # ========== FUNCIÓN PRINCIPAL ========== #   
    def run(self):
        """Ejecuta el sistema completo"""
        if not self.init_camera():
            return
        
        print("\n FLUJO AERODINÁMICO COMPLETO AUTOMATIZADO - TFG UNIFICADO")
        print("="*80)
        print("Flujo automático completo sin intervención manual:")
        print("   1️ Medición AoA con ZED IMU")
        print("   2️ Detección de contornos con YOLO")
        print("   3️ Identificación automática de perfil NACA")
        print("   4️ Análisis XFOIL automático")
        print("   5️ Predicción de campos fluidodinámicos con Deep Learning")
        print("\n Controles:")
        print("   • Modo AoA: R=referencia, S=guardar, D=detección")
        print("   • Modo Detección: S=guardar contornos, A=volver AoA")
        print("   • Q=salir en cualquier modo")
        print("="*70)
        
        try:
            while True:
                if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                    self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                    frame = self.image.get_data()
                    
                    if self.mode == "aoa":
                        # Modo medición AoA
                        if self.is_reference_set:
                            raw_angle = self.get_angle()
                            angle_diff = raw_angle - self.reference_angle
                            
                            while angle_diff > 180:
                                angle_diff -= 360
                            while angle_diff <= -180:
                                angle_diff += 360
                                
                            self.current_angle = angle_diff
                        
                        self.draw_aoa_overlay(frame)
                        cv2.imshow("Flujo Aerodinámico Completo", cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))
                        
                    elif self.mode == "detection":
                        # Modo detección de contornos
                        if self.model is None:
                            if not self.load_yolo_model():
                                print("No se pudo cargar el modelo YOLO. Volviendo a modo AoA...")
                                self.mode = "aoa"
                                continue
                        
                        annotated_frame, contours = self.detect_contours(frame)
                        self.draw_detection_overlay(annotated_frame, len(contours))
                        cv2.imshow("Flujo Aerodinámico Completo", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                        
                        # Guardar contornos temporalmente
                        self.current_contours = contours
                
                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Salir
                    break
                    
                elif self.mode == "aoa":
                    if key == ord('r'):
                        self.set_reference()
                    elif key == ord('s'):
                        self.save_angle()
                    elif key == ord('d'):
                        print("Cambiando a modo detección de contornos...")
                        self.mode = "detection"
                        
                elif self.mode == "detection":
                    if key == ord('s'):
                        if hasattr(self, 'current_contours'):
                            self.save_contours(self.current_contours)
                    elif key == ord('a'):
                        print("Volviendo a modo AoA...")
                        self.mode = "aoa"
                        
        except KeyboardInterrupt:
            print("\n Interrumpido por el usuario")
        finally:
            cv2.destroyAllWindows()
            self.zed.close()
            
            # ========== FLUJO AUTOMÁTICO COMPLETO ==========
            if self.contours_saved:
                print("\n" + "🔬" + "="*68)
                print("INICIANDO FLUJO DE ANÁLISIS AUTOMÁTICO COMPLETO")
                print("🔬" + "="*68)
                
                # Paso 1: Análisis de contornos e identificación NACA
                naca_identificado = self.ejecutar_analisis_contornos()
                
                if naca_identificado:
                    # Paso 2: Análisis XFOIL automático
                    print(f"\n Continuando con análisis XFOIL para NACA {naca_identificado}...")
                    exito_xfoil = self.ejecutar_analisis_xfoil_automatico(naca_identificado)
                    
                    if exito_xfoil:
                        # Paso 3: Análisis de campos fluidodinámicos automático
                        print(f"\n Ejecutando análisis de campos fluidodinámicos para NACA {naca_identificado} @ {self.ultimo_aoa_calculado}°...")
                        exito_campos = self.ejecutar_analisis_campos_fluidodinamicos()
                        
                        print("\n" + "🎉" + "="*68)
                        print("🎉 FLUJO COMPLETO DE TFG TERMINADO EXITOSAMENTE")
                        print("🎉" + "="*68)
                        print("📋 Archivos generados:")
                        print(f"   • aoa_mediciones.txt - Mediciones de ángulo de ataque")
                        print(f"   • contornos.dat - Contornos detectados")
                        print(f"   • NACA{naca_identificado}.dat - Coordenadas del perfil identificado")
                        print(f"   • xfoil_data_{naca_identificado}.csv - Resultados del análisis XFOIL")
                        print(f"   • analisis_completo_NACA_{naca_identificado}.png - Gráficas del análisis XFOIL")
                        if exito_campos:
                            print(f"   • campos_fluidodinamicos_NACA_{naca_identificado}_AoA_{self.ultimo_aoa_calculado:.1f}.png - Campos fluidodinámicos")
                    else:
                        print("\n El análisis XFOIL falló")
                else:
                    print("\n No se pudo identificar el perfil NACA")
            
            # Resumen final de AoA
            if len(self.saved_angles) > 0:
                print(f"\n RESUMEN MEDICIONES AoA:")
                for timestamp, total_angle in self.saved_angles:
                    print(f"   {timestamp}: Ángulo Total = {total_angle:.2f}°")
                print(f"Total: {len(self.saved_angles)} mediciones guardadas")
            
            # Limpiar archivos temporales
            self.limpiar_archivos_temporales()

if __name__ == "__main__":
    print("Iniciando Flujo Aerodinámico Completo Automatizado...")
    sistema = FlujoAerodinamicoCompleto()
    sistema.run()