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
        print(f"⬇️ Descargando/Cargando ECG Real: Record {record_name}...")
        try:
            samp_end = int(duration_sec * 360)
            # Cargar señal
            record = wfdb.rdrecord(record_name, sampfrom=0, sampto=samp_end, pn_dir='mitdb')
            signal = record.p_signal[:, 0] 
            fs = record.fs
            
            # Cargar anotaciones (Solo para validación, no para generar spikes)
            try:
                annotation = wfdb.rdann(record_name, 'atr', sampfrom=0, sampto=samp_end, pn_dir='mitdb')
            except:
                annotation = None
                
            return signal, annotation, fs
            
        except Exception as e:
            print(f"❌ Error cargando MIT-BIH: {e}")
            return None, None, None

    def detect_r_peaks(self, signal, fs):
        """
        Detecta picos usando ENERGÍA (Cuadrado), para pillar tanto positivos como negativos.
        """
        # 1. Eliminar DC (Centrar)
        centered_signal = signal - np.mean(signal)
        
        # 2. Energía (Elevar al cuadrado resalta los picos y vuelve todo positivo)
        squared_signal = centered_signal ** 2
        
        # 3. Normalizar
        sig_min = np.min(squared_signal)
        sig_max = np.max(squared_signal)
        if sig_max - sig_min == 0: return []
        normalized = (squared_signal - sig_min) / (sig_max - sig_min)
        
        # 4. Buscar picos (Umbral 0.25 suele funcionar bien para QRS)
        peaks, _ = find_peaks(normalized, height=0.20, distance=int(fs*0.25))
        
        return peaks

    def measure_qrs_width(self, signal, peak_idx, fs):
        """
        Mide la anchura del latido (en segundos) usando el método FWHM
        (Full Width at Half Maximum) sobre el valor absoluto.
        """
        # Trabajamos con valor absoluto para gestionar picos invertidos (comunes en PVCs)
        abs_sig = np.abs(signal - np.mean(signal))
        
        # Aseguramos límites
        if peak_idx <= 0 or peak_idx >= len(signal) - 1:
            return 0.1 # Valor por defecto seguro
            
        # 1. Altura del pico detectado
        peak_amp = abs_sig[peak_idx]
        
        # 2. Umbral a media altura (50%)
        half_height = peak_amp * 0.5
        
        # 3. Buscar límite izquierdo (inicio del QRS)
        left = peak_idx
        while left > 0 and abs_sig[left] > half_height:
            left -= 1
            
        # 4. Buscar límite derecho (fin del QRS)
        right = peak_idx
        while right < len(signal) - 1 and abs_sig[right] > half_height:
            right += 1
            
        # 5. Calcular ancho en segundos
        width_samples = right - left
        width_seconds = width_samples / fs
        
        return width_seconds

    def ecg_to_spikes(self, record_name='100', duration_sec=5, use_annotations=False):
        """
        Convierte ECG a spikes basándose en la FÍSICA de la señal.
        QRS Ancho -> Jitter Alto
        QRS Estrecho -> Jitter Bajo
        """
        signal, annotation, fs = self.load_mit_bih_data(record_name, duration_sec)
        if signal is None: return np.array([]), np.array([]) * b2.ms

        # 1. Detectar Latidos
        if use_annotations and annotation is not None:
            print("   📝 Usando anotaciones médicas (Ground Truth)")
            beat_indices = annotation.sample
            beat_times_sec = beat_indices / fs
        else:
            print("   🕵️‍♂️ Usando algoritmo de Energía (Autónomo)")
            beat_indices = self.detect_r_peaks(signal, fs)
            beat_times_sec = beat_indices / fs

        print(f"   ❤️ Latidos detectados: {len(beat_times_sec)}")

        # 2. Generación de Spikes con Jitter Dinámico
        indices = []
        times = []
        neurons_pattern = 40 
        
        # Factor de conversión: Ancho real (s) -> Jitter (sigma)
        # Un ancho de 0.10s (ancho) debería dar un jitter notable (~0.03s)
        # Un ancho de 0.04s (estrecho) debería dar un jitter mínimo (~0.005s)
        scaling_factor = 0.4 

        for i, t_beat in enumerate(beat_times_sec):
            idx = beat_indices[i]
            
            # --- FÍSICA: Medir el ancho real del latido ---
            qrs_width = self.measure_qrs_width(signal, idx, fs)
            
            # Calculamos el jitter proporcional
            sigma_jitter = qrs_width * scaling_factor
            
            # Limites de seguridad para que no sea ni 0 ni infinito
            sigma_jitter = np.clip(sigma_jitter, 0.002, 0.050)
            
            # Logging para ver qué está pasando (solo primeros latidos)
            if i < 3:
                tipo = "ANCHO/ARRITMIA" if qrs_width > 0.08 else "ESTRECHO/SANO"
                print(f"      📍 Latido {i}: Ancho={qrs_width*1000:.1f}ms -> Jitter={sigma_jitter*1000:.1f}ms [{tipo}]")

            # Generar los 40 spikes para este latido
            for n_idx in range(neurons_pattern):
                spike_time = t_beat + np.random.normal(0, sigma_jitter)
                if 0 < spike_time < duration_sec:
                    indices.append(n_idx)
                    times.append(spike_time)

        # 3. Ruido de Fondo (SNN no vive en el vacío)
        n_noise = int(self.n_input * duration_sec * 0.5) # 0.5 Hz de ruido base
        indices.extend(np.random.randint(0, self.n_input, n_noise))
        times.extend(np.random.uniform(0, duration_sec, n_noise))

        # 4. Formatear y Ordenar (Obligatorio para Brian2)
        all_indices = np.array(indices, dtype=int)
        all_times = np.array(times) * b2.second
        sort_idx = np.argsort(all_times)
        
        return all_indices[sort_idx], all_times[sort_idx]

    def validate_detection(self, my_times_b2, record_name):
        import wfdb
        try:
            # Quitar unidades si las tiene
            try:
                my_times = np.array(my_times_b2 / b2.second)
            except:
                my_times = np.array(my_times_b2)
            
            # Cargar Ground Truth (médico)
            annotation = wfdb.rdann(record_name, 'atr', sampfrom=0, sampto=360*60, pn_dir='mitdb')
            real_times = annotation.sample / 360.0
            max_sim_time = np.max(my_times) if len(my_times) > 0 else 10
            real_times = real_times[real_times <= max_sim_time]
            
            # --- CLUSTERING INTELIGENTE ---
            my_beats = []
            if len(my_times) > 0:
                # Ordenar tiempos
                my_times = np.sort(my_times)
                
                current_cluster = [my_times[0]]
                for t in my_times[1:]:
                    if t - current_cluster[-1] < 0.050: # Si están cerca (<50ms)
                        current_cluster.append(t)
                    else:
                        # FIN DEL CLUSTER ANTERIOR
                        # FILTRO: Un latido real debe tener al menos 5 spikes (ignoramos ruido suelto)
                        if len(current_cluster) >= 5: 
                            my_beats.append(np.mean(current_cluster))
                        
                        current_cluster = [t] # Empezamos nuevo cluster
                
                # Chequear el último
                if len(current_cluster) >= 5:
                    my_beats.append(np.mean(current_cluster))
            
            print(f"\n   ⚖️  VALIDACIÓN DE ETIQUETADO (Ground Truth):")
            print(f"       - Latidos Médicos: {len(real_times)}")
            print(f"       - Latidos SNN (Filtrados >5 spikes): {len(my_beats)}")
            
            # Calcular precisión
            hits = 0
            for t_real in real_times:
                dist = np.min(np.abs(np.array(my_beats) - t_real)) if len(my_beats) > 0 else 999
                if dist < 0.1: # Margen de error 100ms
                    hits += 1
            
            accuracy = (hits / len(real_times)) * 100 if len(real_times) > 0 else 0
            print(f"       - Coincidencia Temporal: {accuracy:.1f}%")
            
        except Exception as e:
            print(f"       ⚠️ No se pudo validar: {e}")