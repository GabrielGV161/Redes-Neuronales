import numpy as np
import brian2 as b2

def generate_ecg_data(n_input=90, duration_ms=200, bpm=75, mode='healthy', noise_level=0.05):
    """
    Genera spikes sintéticos y LIMPIA duplicados para evitar errores de Brian2.
    """
    # 1. Tiempos base
    interval_sec = 60.0 / bpm
    duration_sec = duration_ms / 1000.0
    beat_times = np.arange(0.1, duration_sec, interval_sec)
    
    indices = []
    times = []
    
    # Rango de neuronas (Mismo cable para ambos modos para hacerlo difícil)
    signal_neurons = np.arange(0, 40) 

    # 2. Generación de Patrones
    if mode == 'healthy':
        sigma_jitter = 0.005 
        for t_beat in beat_times:
            for neuron_idx in signal_neurons:
                spike_t = t_beat + np.random.normal(0, sigma_jitter)
                indices.append(neuron_idx)
                times.append(spike_t)

    elif mode == 'arrhythmia':
        sigma_jitter = 0.030 
        for t_beat in beat_times:
            if np.random.rand() > 0.2: # A veces bloquea
                for neuron_idx in signal_neurons:
                    spike_t = t_beat + np.random.normal(0, sigma_jitter)
                    # Añadir desfase aleatorio
                    spike_t += np.random.uniform(0, 0.050)
                    indices.append(neuron_idx)
                    times.append(spike_t)

    # 3. Ruido
    num_noise = int(n_input * duration_sec * noise_level * 100)
    indices.extend(np.random.randint(0, n_input, num_noise))
    times.extend(np.random.uniform(0, duration_sec, num_noise))
    
    # =========================================================
    # 🧹 LIMPIEZA CRÍTICA (Evita el ValueError de Brian2)
    # =========================================================
    all_indices = np.array(indices, dtype=int)
    all_times = np.array(times)
    
    # 1. Eliminar tiempos fuera de rango (negativos o > duracion)
    mask_valid = (all_times >= 0) & (all_times < duration_sec)
    all_indices = all_indices[mask_valid]
    all_times = all_times[mask_valid]
    
    # 2. Ordenar cronológicamente (Brian2 lo exige)
    sort_idx = np.argsort(all_times)
    all_indices = all_indices[sort_idx]
    all_times = all_times[sort_idx]
    
    # 3. Eliminar duplicados exactos en el mismo dt
    # Brian2 tiene dt=0.1ms. Si dos spikes caen en el mismo bin, crash.
    # Truco: Convertimos a "pasos de tiempo" enteros y buscamos duplicados
    dt = 0.0001 # 0.1 ms
    time_steps = (all_times / dt).astype(int)
    
    # Creamos un identificador único: neuron_id * gran_numero + time_step
    # Esto nos permite encontrar si (neurona 5, tiempo 100) está repe.
    unique_id = all_indices * 1e9 + time_steps
    
    _, unique_idx = np.unique(unique_id, return_index=True)
    
    # Reordenamos porque np.unique devuelve ordenado por valor, no por índice original
    final_indices = all_indices[np.sort(unique_idx)]
    final_times = all_times[np.sort(unique_idx)] * b2.second
    
    return final_indices, final_times