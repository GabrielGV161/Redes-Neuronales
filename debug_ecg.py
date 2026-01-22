# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:45:03 2026

@author: ggv16
"""

import matplotlib.pyplot as plt
import numpy as np
from src.real_ecg_loader2 import RealECGLoader

def debug_detection(record_name='100', duration=5):
    print(f"🔬 DIAGNÓSTICO DEL RECORD {record_name} ({duration}s)")
    
    loader = RealECGLoader()
    
    # 1. Cargar Datos Crudos
    # Usamos la función interna para obtener la señal analógica
    signal, annotation, fs = loader.load_mit_bih_data(record_name, duration)
    
    if signal is None: return

    # 2. Obtener la verdad de los médicos (Anotaciones)
    # Convertimos muestras a segundos para plotear
    ann_samples = annotation.sample
    # Filtramos solo las que entran en la duración
    ann_samples = ann_samples[ann_samples < len(signal)]
    ann_times = ann_samples / fs
    ann_y = [np.max(signal)] * len(ann_times) # Para pintarlos arriba del todo

    # 3. Obtener la detección de tu algoritmo (Autónomo)
    # Usamos tu función detect_r_peaks
    my_peaks_indices = loader.detect_r_peaks(signal, fs)
    my_peaks_times = my_peaks_indices / fs
    my_peaks_y = [np.max(signal) * 0.9] * len(my_peaks_times) # Un poco más abajo

    # 4. PLOTEAR LA VERDAD
    plt.figure(figsize=(12, 6))
    
    # A) La señal analógica
    time_axis = np.arange(len(signal)) / fs
    plt.plot(time_axis, signal, label='Señal ECG (Raw)', color='blue', alpha=0.5, linewidth=1)
    
    # B) Los médicos (Verde)
    plt.scatter(ann_times, signal[ann_samples], color='green', s=100, marker='o', 
                label='Médicos (.atr)', facecolors='none', edgecolors='green', linewidth=2)
    
    # C) Tu Algoritmo (Rojo)
    plt.scatter(my_peaks_times, signal[my_peaks_indices], color='red', s=50, marker='x', 
                label='Tu Algoritmo (Voltaje)')
    
    plt.title(f"Comparativa: Médicos vs Algoritmo (Record {record_name})")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Voltaje (mV)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"   ✅ Latidos Médicos: {len(ann_times)}")
    print(f"   🤖 Latidos Tuyos:   {len(my_peaks_times)}")
    
    if len(ann_times) == len(my_peaks_times):
        print("   🎉 ¡COINCIDENCIA PERFECTA!")
    else:
        print("   ⚠️ DISCREPANCIA DETECTADA (Revisar Umbral)")

if __name__ == "__main__":
    # Prueba con el record normal
    debug_detection('105', duration=5)
    
    # Prueba con el record de arritmia (suele ser más difícil)
    debug_detection('201', duration=5)