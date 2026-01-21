import numpy as np
import brian2 as b2

def generate_ecg_data(n_input=90, duration_ms=10000, bpm=75, noise_level=0.1, mode='healthy'):
    """
    Genera trenes de spikes simulando un ECG.
    
    Args:
        n_input: Número de fibras de entrada (neuronas).
        duration_ms: Duración de la simulación.
        bpm: Pulsaciones por minuto (ritmo base).
        noise_level: Probabilidad de ruido aleatorio (0.0 a 1.0).
        mode: 'healthy' (rítmico) o 'arrhythmia' (caótico).
    """
    dt = 1.0 * b2.ms # Resolución temporal
    duration_sec = duration_ms / 1000
    total_steps = int(duration_ms)
    
    indices = []
    times = []
    
    # 1. GENERAR LATIDOS (SEÑAL)
    # Un corazón sano late cada X ms (aprox 800ms para 75 BPM)
    beat_interval_ms = (60 / bpm) * 1000
    
    current_time = 0
    while current_time < duration_ms:
        # A) Determinar cuándo ocurre el siguiente latido
        if mode == 'healthy':
            # Ritmo regular con variabilidad natural mínima (Jitter biológico)
            interval = np.random.normal(beat_interval_ms, 10) # +/- 10ms
        else:
            # Arritmia (Fibrilación): Intervalos muy caóticos
            # A veces rápido (300ms), a veces lento (1200ms)
            interval = np.random.uniform(300, 1200)
            
        current_time += interval
        if current_time >= duration_ms: break
        
        # B) Generar el "Complejo QRS" (El latido en sí)
        # Un latido es un disparo SINCRONIZADO de muchas neuronas
        # Hacemos que disparen las neuronas 0 a 40 (el patrón a reconocer)
        pattern_neurons = 40 
        
        for i in range(pattern_neurons):
            # No todas disparan al instante exacto, hay una dispersión de ~5ms
            spike_time = current_time + np.random.normal(0, 2)
            indices.append(i)
            times.append(spike_time)
            
    # 2. GENERAR RUIDO (ARTEFACTOS)
    # El ruido afecta a TODAS las neuronas (0 a 89) de forma aleatoria
    # Es ruido de fondo continuo (Poisson)
    
    noise_rate = noise_level * 50 # Hz de ruido base
    n_noise_spikes = int(noise_rate * duration_sec * n_input)
    
    noise_indices = np.random.randint(0, n_input, n_noise_spikes)
    noise_times = np.random.uniform(0, duration_ms, n_noise_spikes)
    
    indices.extend(noise_indices)
    times.extend(noise_times)
    
    # 3. EMPAQUETAR Y ORDENAR
    all_indices = np.array(indices, dtype=int)
    all_times = np.array(times) * b2.ms
    
    # Ordenar cronológicamente (Brian2 lo exige)
    sort_idx = np.argsort(all_times)
    
    print(f"🫀 ECG Generado ({mode}): {int(bpm)} BPM aprox")
    print(f"   - Señal: Neuronas 0-40 (Sincronizadas)")
    print(f"   - Ruido: Todo el canal (Nivel {noise_level})")
    
    return all_indices[sort_idx], all_times[sort_idx]