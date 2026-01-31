import wfdb
import numpy as np
import brian2 as b2
import os

class RealECGLoader:
    def __init__(self, n_input=90):
        self.n_input = n_input
        
    def load_mit_bih_data(self, record_name='100', duration_sec=10):
        """
        Descarga y carga datos reales del MIT-BIH Arrhythmia Database.
        """
        print(f"⬇️ Descargando/Cargando ECG Real: Record {record_name}...")
        
        try:
            # CORRECCIÓN: Cambiado 'pb_dir' por 'pn_dir' para versiones nuevas de wfdb
            # sampfrom y sampto definen qué trozo leemos (frecuencia 360Hz)
            samp_end = int(duration_sec * 360)
            
            # Leer señal (pn_dir='mitdb' descarga de PhysioNet automáticamente)
            record = wfdb.rdrecord(record_name, sampfrom=0, sampto=samp_end, pn_dir='mitdb')
            
            # Leer anotaciones (los picos reales marcados por médicos)
            annotation = wfdb.rdann(record_name, 'atr', sampfrom=0, sampto=samp_end, pn_dir='mitdb')
            
        except Exception as e:
            print(f"❌ Error cargando PhysioNet: {e}")
            print("   (Verifica tu conexión a internet o intenta descargar los archivos '100.dat', '100.hea', '100.atr' manualmente)")
            return None, None, None

        signal = record.p_signal[:, 0] # Usamos la derivación MLII
        fs = record.fs # Frecuencia de muestreo
        
        print(f"✅ Datos cargados: {duration_sec}s a {fs}Hz")
        return signal, annotation, fs

    def ecg_to_spikes(self, record_name='100', duration_sec=5):
        """
        Convierte el ECG analógico en trenes de spikes.
        """
        signal, annotation, fs = self.load_mit_bih_data(record_name, duration_sec)
        
        # Si falló la carga, devolvemos vacío para no crashear
        if signal is None: 
            return np.array([]), np.array([]) * b2.ms

        indices = []
        times = []

        # 1. OBTENER MOMENTOS DE LATIDOS
        beat_samples = annotation.sample
        beat_times_sec = beat_samples / fs

        print(f"❤️ Latidos reales detectados: {len(beat_times_sec)}")

        # 2. GENERAR SPIKES (Neuromorphic Encoding)
        # Neuronas 0-40 disparan cuando hay latido
        neurons_pattern = 40 
        jitter_window = 0.005 

        for t_beat in beat_times_sec:
            for i in range(neurons_pattern):
                spike_time = t_beat + np.random.normal(0, jitter_window)
                if 0 < spike_time < duration_sec:
                    indices.append(i)
                    times.append(spike_time)

        # 3. AÑADIR RUIDO DE FONDO
        # Neuronas 40-90 tienen ruido
        n_noise = int(100 * duration_sec)
        indices.extend(np.random.randint(40, self.n_input, n_noise))
        times.extend(np.random.uniform(0, duration_sec, n_noise))

        # Ordenar
        all_indices = np.array(indices, dtype=int)
        all_times = np.array(times) * b2.second
        sort_idx = np.argsort(all_times)

        return all_indices[sort_idx], all_times[sort_idx]

# Helper rápido para testear
if __name__ == "__main__":
    loader = RealECGLoader()
    print("--- Testeando carga de datos reales ---")
    i, t = loader.ecg_to_spikes(record_name='200', duration_sec=5)
    
    if len(i) > 0:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,4))
        plt.plot(t, i, '.k', markersize=2)
        plt.title("Raster Plot de ECG REAL (MIT-BIH 200)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Neurona ID")
        plt.show()
    else:
        print("No se pudieron cargar los datos.")