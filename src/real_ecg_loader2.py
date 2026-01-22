# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:30:36 2026

@author: ggv16
"""

import wfdb
import numpy as np
import brian2 as b2
from scipy.signal import find_peaks

class RealECGLoader:
    def __init__(self, n_input=90):
        self.n_input = n_input
        
    def load_mit_bih_data(self, record_name='100', duration_sec=10):
        # (Esta función se queda igual que la anterior)
        print(f"⬇️ Descargando/Cargando ECG Real: Record {record_name}...")
        try:
            samp_end = int(duration_sec * 360)
            record = wfdb.rdrecord(record_name, sampfrom=0, sampto=samp_end, pn_dir='mitdb')
            # Intentamos cargar anotaciones, pero si fallan no pasa nada
            try:
                annotation = wfdb.rdann(record_name, 'atr', sampfrom=0, sampto=samp_end, pn_dir='mitdb')
            except:
                annotation = None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None, None

        signal = record.p_signal[:, 0] 
        fs = record.fs 
        return signal, annotation, fs

    def detect_r_peaks(self, signal, fs):
        """
        🕵️‍♂️ DETECTIVE DE PICOS MEJORADO (Energía):
        Detecta latidos positivos (normales) Y negativos (arritmias)
        usando la energía de la señal (cuadrado).
        """
        # 1. Centrar la señal en 0 (eliminar offset DC)
        # A veces la señal flota en 0.5mV, la bajamos a 0.
        centered_signal = signal - np.mean(signal)
        
        # 2. ELEVAR AL CUADRADO (La Clave 🔑)
        # Esto convierte los picos negativos profundos (Arritmias) en positivos.
        # También penaliza el ruido pequeño y amplifica los picos grandes.
        squared_signal = centered_signal ** 2
        
        # 3. Normalizar la señal cuadrada entre 0 y 1
        sig_min = np.min(squared_signal)
        sig_max = np.max(squared_signal)
        
        if sig_max - sig_min == 0: return []
        
        normalized_signal = (squared_signal - sig_min) / (sig_max - sig_min)
        
        # 4. BUSCAR PICOS
        # Ajustamos parámetros:
        # - height=0.25: Al elevar al cuadrado, los picos destacan mucho más, 
        #   así que podemos bajar el umbral relativo para no perdernos ninguno.
        # - distance=fs*0.25: Refractario de 250ms.
        peaks, _ = find_peaks(normalized_signal, height=0.25, distance=int(fs*0.25))
        
        return peaks

    def ecg_to_spikes(self, record_name='100', duration_sec=5, use_annotations=False):
        """
        Convierte ECG a spikes.
        Args:
            use_annotations (bool): 
                True  -> Usa la "verdad absoluta" de los médicos (.atr).
                False -> Usa nuestro algoritmo de detección (Realista/Autónomo).
        """
        signal, annotation, fs = self.load_mit_bih_data(record_name, duration_sec)
        if signal is None: return np.array([]), np.array([]) * b2.ms

        beat_times_sec = []

        # --- CAMINO A: TRAMPA (Usar médicos) ---
        if use_annotations and annotation is not None:
            print("📝 Usando anotaciones médicas (Ground Truth)")
            beat_times_sec = annotation.sample / fs

        # --- CAMINO B: REALISTA (Usar voltaje) ---
        else:
            print("🕵️‍♂️ Usando algoritmo de detección de voltaje (Autónomo)")
            beat_indices = self.detect_r_peaks(signal, fs)
            beat_times_sec = beat_indices / fs

        print(f"❤️ Latidos detectados: {len(beat_times_sec)}")

        # GENERACIÓN DE SPIKES (Igual que antes)
        indices = []
        times = []
        neurons_pattern = 40 
        jitter_window = 0.005 

        for t_beat in beat_times_sec:
            for i in range(neurons_pattern):
                spike_time = t_beat + np.random.normal(0, jitter_window)
                if 0 < spike_time < duration_sec:
                    indices.append(i)
                    times.append(spike_time)

        # Ruido de fondo
        n_noise = int(100 * duration_sec)
        indices.extend(np.random.randint(40, self.n_input, n_noise))
        times.extend(np.random.uniform(0, duration_sec, n_noise))

        all_indices = np.array(indices, dtype=int)
        all_times = np.array(times) * b2.second
        sort_idx = np.argsort(all_times)

        return all_indices[sort_idx], all_times[sort_idx]

# Helper para comparar visualmente
if __name__ == "__main__":
    loader = RealECGLoader()
    
    # Prueba autónoma
    print("\n--- MODO AUTÓNOMO ---")
    i, t = loader.ecg_to_spikes(record_name='100', duration_sec=5, use_annotations=False)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.plot(t, i, '.k', markersize=2)
    plt.title("ECG Real detectado por VOLTAJE (Sin anotaciones)")
    plt.xlabel("Tiempo (s)")
    plt.show()