
import pandas as pd
from glob import glob
import os
import numpy as np
import torch

def cargar_datos(dirc):
    archivos = glob(os.path.join(dirc, '*'), recursive=True)
    archivos = [f for f in archivos if os.path.isfile(f)]


    ys = []  # Salidas
    xs = []  # Entradas

    for archivo in archivos:
        name = os.path.basename(archivo)
        name = name.replace("naca","").split("_")

        # GEOMETRIA 
        naca_str = name[0]  
        NACA = np.float32(naca_str)
        p = int(naca_str[1]) / 10.0   # Posición del máximo de curvatura
        t = int(naca_str[2:]) / 100.0 # Espesor máximo

        # AoA (tiene en cuenta decimales y numeros enteros)
        try:
            parte_entera = name[1]
            if len(name) > 2 and name[2].split('.')[0].lstrip('-').isdigit():
                parte_decimal = name[2].split('.')[0]
                AoA = np.float32(f"{parte_entera}.{parte_decimal}")
            else:
                AoA = np.float32(parte_entera)
        except Exception as e:
            print(f"Error extrayendo AoA de: {name} -> {e}")
            AoA = 0.0

        # Guardar entrada
        xs.append([p, t, AoA])

        csv = pd.read_csv(archivo).to_numpy()[0:25000,1:5].transpose()
        ys.append(csv)

    y = np.stack(ys, axis=0)
    x = np.array(xs, dtype=np.float32)
    return x, y

def custom_loss(y_pred, y_true):
    # Separar canales
    xy_pred = y_pred[:, 0:2, :]
    xy_true = y_true[:, 0:2, :]
    p_pred = y_pred[:, 2, :]
    p_true = y_true[:, 2, :]
    v_pred = y_pred[:, 3, :]
    v_true = y_true[:, 3, :]

    # MSE para coordenadas
    loss_xy = torch.mean((xy_pred - xy_true) ** 2)
    # MSE para presión
    loss_p = torch.mean((p_pred - p_true) ** 2)
    # MSE para velocidad
    loss_v = torch.mean((v_pred - v_true) ** 2)
    # Penalización para presión y velocidad (no para coordenadas)
    eps = 1e-8  # Para evitar división por cero
    penal_p = torch.abs(p_pred) / (torch.abs(p_true) + eps) >= 2
    penal_v = torch.abs(v_pred) / (torch.abs(v_true) + eps) >= 2

    penalty = 0.0
    if penal_p.any():
        penalty += torch.mean((p_pred[penal_p] - p_true[penal_p]) ** 2) * 10  # penaliza x10
    if penal_v.any():
        penalty += torch.mean((v_pred[penal_v] - v_true[penal_v]) ** 2) * 10

    return loss_xy + loss_p + loss_v + penalty

def custom_loss_xy(y_pred, y_true):
    # y_pred, y_true: (batch, 2, puntos)
    # MSE para coordenadas
    loss_xy = torch.mean((y_pred - y_true) ** 2)
    return loss_xy

def custom_loss_single(y_pred, y_true):
    # y_pred, y_true: (batch, 1, puntos)
    # MSE para el canal
    loss = torch.mean((y_pred - y_true) ** 2)
    # Penalización para errores mayores al 200%
    eps = 1e-8
    penal = torch.abs(y_pred) / (torch.abs(y_true) + eps) >= 2
    penalty = 0.0
    if penal.any():
        penalty += torch.mean((y_pred[penal] - y_true[penal]) ** 2) * 10
    return loss + penalty

import glob
import os
import numpy as np
from PIL import Image

def cargar_imagenes_2d(directorio, tipos=('pressure', 'velocity-magnitude'), shape=(512, 512)):
    
    archivos = []
    for tipo in tipos:
        archivos += sorted(glob.glob(os.path.join(directorio, f"*_{tipo}.png")))

    muestras = {}
    for archivo in archivos:
        base = os.path.basename(archivo)
        nombre_base = "_".join(base.split("_")[:2])
        tipo = base.split("_")[-1].replace(".png", "")
        if nombre_base not in muestras:
            muestras[nombre_base] = {}
        muestras[nombre_base][tipo] = archivo

    X = []
    Y = []
    nombres = []
    for nombre_base, tipos_dict in muestras.items():
        canales = []
        name = nombre_base.replace("naca", "").split("_")
        naca_str = name[0]
        p = int(naca_str[1]) / 10.0
        t = int(naca_str[2:]) / 100.0

        # AoA (tiene en cuenta decimales y numeros enteros)
        try:
            parte_entera = name[1]
            if len(name) > 2 and name[2].split('.')[0].lstrip('-').isdigit():
                parte_decimal = name[2].split('.')[0]
                AoA = np.float32(f"{parte_entera}.{parte_decimal}")
            else:
                AoA = np.float32(parte_entera)
        except Exception as e:
            print(f"Error extrayendo AoA de: {name} -> {e}")
            AoA = 0.0

        X.append([p, t, AoA])
        for tipo in tipos:
            if tipo in tipos_dict:
                img = Image.open(tipos_dict[tipo]).convert('F').resize(shape)
                canales.append(np.array(img))
            else:
                canales.append(np.zeros(shape))
        Y.append(np.stack(canales, axis=0))
        nombres.append(nombre_base)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return X, Y, nombres

import torch.nn.functional as F

def gradient_loss(y_pred, y_true):
    # Calcula diferencias en x e y (bordes)
    dx_pred = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]
    dy_pred = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    dx_true = y_true[:, :, :, 1:] - y_true[:, :, :, :-1]
    dy_true = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
    loss_x = F.l1_loss(dx_pred, dx_true)
    loss_y = F.l1_loss(dy_pred, dy_true)
    return loss_x + loss_y

def cargar_campos_npy(directorio, tipos=('pressure', 'velocity-magnitude'), shape=(512, 512)):
    import glob, os
    X = []
    Y = []
    nombres = []
    archivos = []
    for tipo in tipos:
        archivos += sorted(glob.glob(os.path.join(directorio, f"*_{tipo}.npy")))

    muestras = {}
    for archivo in archivos:
        base = os.path.basename(archivo)
        nombre_base = "_".join(base.split("_")[:2])
        tipo = base.split("_")[-1].replace(".npy", "")
        if nombre_base not in muestras:
            muestras[nombre_base] = {}
        muestras[nombre_base][tipo] = archivo

    for nombre_base, tipos_dict in muestras.items():
        canales = []
        name = nombre_base.replace("naca", "").split("_")
        naca_str = name[0]
        p = int(naca_str[1]) / 10.0
        t = int(naca_str[2:]) / 100.0
        try:
            parte_entera = name[1]
            if len(name) > 2 and name[2].split('.')[0].lstrip('-').isdigit():
                parte_decimal = name[2].split('.')[0]
                AoA = np.float32(f"{parte_entera}.{parte_decimal}")
            else:
                AoA = np.float32(parte_entera)
        except Exception as e:
            print(f"Error extrayendo AoA de: {name} -> {e}")
            AoA = 0.0

        X.append([p, t, AoA])
        for tipo in tipos:
            if tipo in tipos_dict:
                campo = np.load(tipos_dict[tipo])
                if campo.shape != shape:
                    campo = np.resize(campo, shape)
                canales.append(campo)
            else:
                canales.append(np.zeros(shape))
        Y.append(np.stack(canales, axis=0))
        nombres.append(nombre_base)
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return X, Y, nombres