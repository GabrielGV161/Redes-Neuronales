# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 11:26:19 2026

@author: ggv16
"""
# src/audio_encoder.py
import numpy as np
import librosa
import brian2 as b2

class AudioToSpikeEncoder:
    """
    Convierte audio en trenes de spikes.
    Para Indiveri: Nos centraremos en la descomposición espectral (Cochlea-like).
    """
    def __init__(self, n_neurons=90, sample_rate=22050):
        # Usamos 90 neuronas para coincidir con tu TOTAL_NEURONAS_INPUT actual
        self.n_neurons = n_neurons
        self.sample_rate = sample_rate
        
    def load_audio(self, filepath, duration_ms=None):
        """Carga y normaliza el audio"""
        duration_sec = duration_ms / 1000 if duration_ms else None
        # Mono=True mezcla canales si es estéreo
        y, sr = librosa.load(filepath, sr=self.sample_rate, duration=duration_sec, mono=True)
        # Normalizar amplitud a [-1, 1]
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        return y

    def spectral_coding(self, audio_data, max_rate=100*b2.Hz, debug=True):
        """
        Simula una cóclea simplificada:
        - Eje Y (Neuronas) = Frecuencia
        - Eje X (Tiempo) = Tiempo
        - Intensidad del pixel = Probabilidad de disparo
        """
        if debug: print(f"🎵 Codificando audio ({len(audio_data)/self.sample_rate:.2f}s) a spikes...")

        # 1. Espectrograma (Short-Time Fourier Transform)
        # Calculamos frecuencias para que coincidan con el número de neuronas
        n_fft = 1024
        hop_length = 256 # Salto temporal (cuanto menor, más resolución temporal)
        
        # stft devuelve matriz compleja [frecuencias, tiempo]
        D = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Convertir dB a magnitud lineal normalizada [0, 1]
        # S_db suele ir de -80dB a 0dB. Lo escalamos.
        S_norm = (S_db + 80) / 80
        S_norm = np.clip(S_norm, 0, 1) # Recortar ruido de fondo
        
        # 2. Redimensionar al número de neuronas (n_neurons = bandas de frecuencia)
        # Usamos una interpolación para ajustar las bandas de frecuencia a tus neuronas
        import scipy.ndimage
        zoom_factor_freq = self.n_neurons / S_norm.shape[0]
        # No cambiamos el tiempo (zoom 1.0), solo frecuencias
        S_resized = scipy.ndimage.zoom(S_norm, (zoom_factor_freq, 1.0), order=1)
        
        # 3. Generación de Spikes (Poisson)
        indices = []
        times = []
        
        dt_frame = hop_length / self.sample_rate # Tiempo que dura cada columna del espectrograma
        
        # Recorremos la matriz [neurona, tiempo]
        rows, cols = S_resized.shape
        
        # Vectorizamos para velocidad (truco pro)
        # Probabilidad de disparo por bin = Tasa * dt
        prob_matrix = S_resized * (max_rate/b2.Hz) * dt_frame
        
        # Generar matriz aleatoria del mismo tamaño
        random_matrix = np.random.rand(rows, cols)
        
        # Donde random < prob, hay spike
        spike_coords = np.where(random_matrix < prob_matrix)
        
        # spike_coords[0] son filas (neuronas), spike_coords[1] son columnas (tiempo)
        indices = spike_coords[0]
        times_frames = spike_coords[1]
        
        # Convertir frames a segundos
        times = times_frames * dt_frame * b2.second
        
        # Ordenar por tiempo (necesario para Brian2)
        sort_idx = np.argsort(times)
        indices = indices[sort_idx]
        times = times[sort_idx]
        
        if debug:
            print(f"✅ Generados {len(indices)} spikes.")
            print(f"   Neuronas activas: {len(np.unique(indices))}/{self.n_neurons}")
        
        return indices, times

# Función helper rápida
def encode_audio_file(filepath, n_neurons=90, max_rate=100*b2.Hz):
    encoder = AudioToSpikeEncoder(n_neurons=n_neurons)
    y = encoder.load_audio(filepath)
    return encoder.spectral_coding(y, max_rate=max_rate)
