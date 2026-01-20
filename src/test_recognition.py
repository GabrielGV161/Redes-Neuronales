# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 22:34:07 2026

@author: ggv16
"""

# -*- coding: utf-8 -*-
"""
Funciones para testear si la red reconoce patrones
"""
import brian2 as b2
import numpy as np

def test_pattern_recognition(network_dict, spike_indices, spike_times, duration, pattern_size):
    """
    Presenta un patrón a la red (sin aprendizaje) y mide la respuesta.
    
    Args:
        network_dict: Red con pesos YA entrenados
        test_data: Tupla (indices, times) del patrón a testear
        duration: Duración del test
        pattern_size: Número de neuronas en el patrón
    
    Returns:
        dict: Métricas de reconocimiento
    """
    # 1. Restaurar estado inicial (pesos y voltajes)
    net = network_dict['net']
    net.restore('initial_state')
    
    # 2. Configurar entrada
    input_gen = network_dict['input']
    input_gen.set_spikes(spike_indices, spike_times)
    
    # 3. Ejecutar simulación
    net.run(duration)
    
   # 4. Recoger datos de monitores
    mon_out = network_dict['mon_out']
    mon_v = network_dict['mon_v']  # El StateMonitor del voltaje
    
# --- ESTADÍSTICAS BÁSICAS ---
    count = mon_out.count
    if len(count) > 0:
        total_spikes = len(mon_out.t)
        active_neurons = len(np.unique(mon_out.i))
        max_response = np.max(count)
        # Tasa promedio en Hz
        mean_rate = total_spikes / network_dict['n_output'] / (duration/b2.second)
    else:
        total_spikes = 0
        active_neurons = 0
        max_response = 0
        mean_rate = 0
        
    print(f"   Spikes totales: {total_spikes}")
    print(f"   Neuronas activas: {active_neurons}/{network_dict['n_output']}")
    print(f"   Tasa promedio: {mean_rate:.1f} Hz")
    
    # --- EXTRAER DATOS PARA GRÁFICA (Sin unidades para Matplotlib) ---
    results = {
        'total_spikes': total_spikes,
        'mean_rate': mean_rate,
        # Datos Input
        'input_indices': np.array(spike_indices),
        'input_times': np.array(spike_times / b2.ms),
        # Datos Output (Voltaje) -> IMPORTANTE: Dividir por b2.mV
        'time_trace': np.array(mon_v.t / b2.ms),
        'voltage_trace': np.array(mon_v.v / b2.mV),
        # Datos Output (Spikes - Raster)
        'output_spikes_t': np.array(mon_out.t / b2.ms),
        'output_spikes_i': np.array(mon_out.i)
    }
    
    return results

def compare_recognition(network_dict, pattern_trained, pattern_novel, duration, pattern_size):
    """
    Compara A vs B y devuelve los resultados crudos para plotear.
    """
    # Guardar estado inicial antes de nada
    network_dict['net'].store('initial_state')
    
    print("\n" + "="*60)
    print(" TEST 1: Patrón entrenado")
    print("="*60)
    indices_A, times_A = pattern_trained
    res_A = test_pattern_recognition(network_dict, indices_A, times_A, duration, pattern_size)
    
    print("\n" + "="*60)
    print(" TEST 2: Patrón novel")
    print("="*60)
    indices_B, times_B = pattern_novel
    res_B = test_pattern_recognition(network_dict, indices_B, times_B, duration, pattern_size)
    
    # Calcular ratios
    rate_A = res_A['mean_rate']
    rate_B = res_B['mean_rate']
    
    ratio = rate_A / rate_B if rate_B > 0 else rate_A # Evitar div por 0
    diff = rate_A - rate_B
    
    print("\n" + "="*60)
    print(" DISCRIMINACIÓN")
    print("="*60)
    print(f"   Ratio (A/B): {ratio:.2f}x")
    print(f"   Diferencia: {diff:+.1f} Hz")
    
    if ratio > 2.0:
        print("   ✅ LA RED DISCRIMINA (Prefiere el patrón entrenado)")
    else:
        print("   ❌ NO DISCRIMINA CLARAMENTE")
        
    # Devolvemos TODO: estadísticas y los datos crudos de A y B
    stats = {'ratio': ratio, 'diff': diff}
    return stats, res_A, res_B