# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 12:30:36 2026

@author: ggv16
"""

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
            # Cargar señal completa (wfdb entrega fs en el objeto)
            record = wfdb.rdrecord(record_name, sampfrom=0, pn_dir='mitdb')
            fs = record.fs
            samp_end = int(duration_sec * fs)
            signal = record.p_signal[:samp_end, 0] 
            
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

    def ecg_to_spikes(self, input_data, fs=360.0, duration_sec=10):
        """
        Convierte ECG a Spikes.
        Args:
            input_data: Puede ser un STR (nombre del record MIT-BIH) o un ARRAY (señal raw).
            fs: Frecuencia de muestreo (solo necesaria si input_data es array).
            duration_sec: Duración a procesar.
        """
        import wfdb
        import numpy as np
        
        try:
            # --- CASO 1: Es un nombre de archivo (String) ---
            if isinstance(input_data, str):
                # Usamos la función interna para cargar y normalizar
                # CORRECCIÓN: load_mit_bih_data devuelve 3 valores (signal, times, annotations)
                # Usamos un comodín (*) para ignorar lo que sobre, o recogemos la tupla.
                loaded_data = self.load_mit_bih_data(input_data, duration_sec)
                
                # Asumimos que el primer elemento es la señal
                signal = loaded_data[0]
                
                # MIT-BIH siempre es 360 Hz (si tu load_mit_bih_data no devuelve fs, usamos 360)
                # Si tu función load_mit_bih_data devolviera fs, ajustaríamos aquí.
                record_fs = 360.0 
                fs = record_fs
                
            # --- CASO 2: Es una señal directa (Array/Numpy) ---
            elif isinstance(input_data, (np.ndarray, list)):
                signal = np.array(input_data)
                # Aquí confiamos en el 'fs' que nos pasan como argumento
                
            else:
                raise ValueError("input_data debe ser nombre de registro (str) o señal (array)")

            # -----------------------------------------------------------
            # A PARTIR DE AQUÍ, EL PROCESO ES IGUAL PARA AMBOS CASOS
            # -----------------------------------------------------------
            
            # 1. Detectar Picos (Algoritmo de Energía)
            peaks = self.detect_r_peaks(signal, fs)
            
            # 2. Medir Ancho (FWHM) y Mapear a Jitter
            indices = []
            times = []
            
            for r_peak in peaks:
                # Ventana de 100ms alrededor del pico
                window_samples = int(0.100 * fs)
                start = max(0, r_peak - window_samples)
                end = min(len(signal), r_peak + window_samples)
                
                beat_window = signal[start:end]
                if len(beat_window) < 5: continue
                
                # Calcular FWHM (Full Width at Half Maximum)
                peak_val = signal[r_peak]
                half_max = peak_val / 2.0
                # Cruces por el valor medio
                crossings = np.where(np.diff(np.sign(beat_window - half_max)))[0]
                
                width_ms = 0
                if len(crossings) >= 2:
                    width_samples = crossings[-1] - crossings[0]
                    width_ms = (width_samples / fs) * 1000
                else:
                    width_ms = 20 # Valor por defecto si falla el cálculo
                
                # --- MAPEO FÍSICO: ANCHO -> JITTER ---
                if width_ms < 40: # QRS Estrecho (Sano)
                    jitter = 5e-3 # 5ms
                else:             # QRS Ancho (Arritmia/PVC)
                    jitter = 25e-3 # 25ms
                
                # Generar 40 spikes para este latido
                for i in range(self.n_input):
                    t_spike = (r_peak / fs) + np.random.normal(0, jitter)
                    if 0 <= t_spike < duration_sec:
                        indices.append(i)
                        times.append(t_spike)
            
            # 3. Ruido de Fondo (Reducido al 5%)
            n_noise = int(self.n_input * duration_sec * 0.05) 
            indices.extend(np.random.randint(0, self.n_input, n_noise))
            times.extend(np.random.uniform(0, duration_sec, n_noise))
            
            # 4. ORDENAMIENTO TEMPORAL ESTRICTO (Fix crítico para Brian2)
            # Quitamos unidades, ordenamos y devolvemos números puros (segundos)
            # para que 'main.py' luego les ponga *b2.second
            all_times = np.array(times)
            sort_idx = np.argsort(all_times)
            
            indices = np.array(indices)[sort_idx]
            times = all_times[sort_idx]
            
            return indices, times
            
        except Exception as e:
            print(f"❌ Error en ecg_to_spikes: {e}")
            return [], []

    def validate_detection(self, my_times_b2, record_name):
        import wfdb
        try:
            # Quitar unidades si las tiene
            try:
                my_times = np.array(my_times_b2 / b2.second)
            except:
                my_times = np.array(my_times_b2)
            
            # Cargar Ground Truth (médico) y frecuencia de muestreo
            record = wfdb.rdrecord(record_name, sampfrom=0, pn_dir='mitdb')
            fs = record.fs
            annotation = wfdb.rdann(record_name, 'atr', sampfrom=0, sampto=360*60, pn_dir='mitdb')
            real_times = annotation.sample / fs
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
                        # FILTRO: Un latido real debe tener al menos 15 spikes (ignoramos ruido suelto)
                        if len(current_cluster) >= 15: 
                            my_beats.append(np.mean(current_cluster))
                        
                        current_cluster = [t] # Empezamos nuevo cluster
                
                # Chequear el último
                if len(current_cluster) >= 15:
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