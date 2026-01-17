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

def test_pattern_recognition(network_dict, test_data, duration, pattern_size):
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
    # Reiniciar monitores
    network_dict['mon_in'].active = True
    network_dict['mon_out'].active = True
    
    # Inyectar nuevo patrón
    indices, times = test_data
    network_dict['input'].set_spikes(indices, times)
    
    # Correr (sin modificar pesos porque learning_enabled=False)
    network_dict['net'].run(duration)
    
    # Analizar respuesta
    output_spikes = network_dict['mon_out']
    
    if len(output_spikes.t) == 0:
        print("⚠️  No hubo respuesta de la red")
        return {
            'total_spikes': 0,
            'active_neurons': 0,
            'spike_rate_hz': 0.0,
            'response_strength': 0.0
        }
    
    # Métricas básicas
    total_spikes = len(output_spikes.t)
    active_neurons = len(np.unique(output_spikes.i))
    duration_sec = float(duration / b2.second)
    spike_rate = total_spikes / duration_sec
    
    # "Fuerza" de respuesta = promedio de spikes por neurona activa
    spikes_per_neuron = np.bincount(output_spikes.i, minlength=network_dict['n_output'])
    response_strength = spikes_per_neuron.max()  # Neurona más activa
    
    results = {
        'total_spikes': int(total_spikes),
        'active_neurons': int(active_neurons),
        'spike_rate_hz': float(spike_rate),
        'response_strength': float(response_strength),
        'spikes_per_neuron': spikes_per_neuron
    }
    
    print(f"\n🔍 Resultados del reconocimiento:")
    print(f"   Spikes totales: {total_spikes}")
    print(f"   Neuronas activas: {active_neurons}/{network_dict['n_output']}")
    print(f"   Tasa promedio: {spike_rate:.1f} Hz")
    print(f"   Respuesta máxima: {response_strength:.0f} spikes")
    
    return results


def compare_recognition(network_dict, pattern_trained, pattern_novel, 
                       duration, pattern_size):
    """
    Compara respuesta a patrón entrenado vs patrón nuevo.
    
    Args:
        network_dict: Red entrenada
        pattern_trained: Datos del patrón con el que se entrenó
        pattern_novel: Datos de un patrón diferente
        duration: Duración de cada test
        pattern_size: Tamaño del patrón
    
    Returns:
        dict: Comparación de métricas
    """
    print("\n" + "="*60)
    print("🧪 TEST 1: Patrón entrenado")
    print("="*60)
    results_trained = test_pattern_recognition(
        network_dict, pattern_trained, duration, pattern_size
    )
    
    # Reiniciar la red para el segundo test
    network_dict['net'].restore()  # Vuelve al estado inicial
    
    print("\n" + "="*60)
    print("🧪 TEST 2: Patrón novel (no visto)")
    print("="*60)
    results_novel = test_pattern_recognition(
        network_dict, pattern_novel, duration, pattern_size
    )
    
    # Calcular discriminación
    discrimination = {
        'spike_ratio': results_trained['total_spikes'] / max(results_novel['total_spikes'], 1),
        'strength_ratio': results_trained['response_strength'] / max(results_novel['response_strength'], 1),
        'rate_difference': results_trained['spike_rate_hz'] - results_novel['spike_rate_hz']
    }
    
    print("\n" + "="*60)
    print("📊 DISCRIMINACIÓN")
    print("="*60)
    print(f"   Ratio de spikes (entrenado/novel): {discrimination['spike_ratio']:.2f}x")
    print(f"   Ratio de fuerza: {discrimination['strength_ratio']:.2f}x")
    print(f"   Diferencia de tasa: {discrimination['rate_difference']:+.1f} Hz")
    
    if discrimination['spike_ratio'] > 1.5:
        print("   ✅ La red DISCRIMINA bien (responde más al patrón entrenado)")
    elif discrimination['spike_ratio'] > 1.1:
        print("   ⚠️  Discriminación débil")
    else:
        print("   ❌ No discrimina (responde igual a ambos)")
    
    return {
        'trained': results_trained,
        'novel': results_novel,
        'discrimination': discrimination
    }