# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 15:19:39 2026

@author: ggv16
"""
import pytest
import numpy as np
import os
import sys
import brian2 as b2

# Truco para que Python encuentre la carpeta 'src' si ejecutas desde 'tests'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.weight_manager import save_weights, load_weights, get_weights_dir
from src.real_ecg_loader import RealECGLoader

# =============================================================================
#  TEST 1: Roundtrip de Pesos (Guardar -> Cargar -> Verificar)
# =============================================================================

def test_weight_manager_roundtrip():
    # 1. Configuración del entorno de prueba (Brian2)
    b2.start_scope()
    
    # --- CORRECCIÓN AQUÍ ---
    # Añadimos threshold='v > 1*volt' y reset='v=0*volt'.
    # Aunque no corramos la simulación, Brian2 lo exige para aceptar la sinapsis.
    G_in = b2.NeuronGroup(10, 'v : volt', threshold='v > 1*volt', reset='v=0*volt', method='exact')
    G_out = b2.NeuronGroup(10, 'v : volt', method='exact')
    
    # Creamos sinapsis
    S = b2.Synapses(G_in, G_out, 'w : volt', on_pre='v += w')
    S.connect(p=0.5)  
    
    # Asignamos valores específicos 
    original_weights = np.random.rand(len(S)) * 10 * b2.mV
    S.w = original_weights
    
    # Empaquetamos 
    network_dict = {'synapses': S}
    test_filename = "test_weights_pytest.pkl"
    
    # 2. Guardar
    print(f"\n[TEST] Guardando pesos de prueba: {test_filename}")
    save_weights(network_dict, filename=test_filename)
    
    # 3. "Corromper" la red actual 
    S.w = 0 * b2.mV
    # Nota: usamos una pequeña tolerancia por si hay errores de coma flotante en la asignación interna
    assert np.all(np.abs(S.w) < 1e-9 * b2.volt), "Error en el setup: pesos no reseteados."
    
    # 4. Cargar
    print("[TEST] Cargando pesos...")
    load_weights(network_dict, filename=test_filename)
    
    # 5. Aserciones
    loaded_weights = S.w
    
    # Quitamos unidades para comparar numéricamente con numpy
    # (Brian2 a veces devuelve array con unidades, a veces no, dependiendo de la versión)
    try:
        loaded_w_no_unit = np.array(loaded_weights / b2.mV)
        orig_w_no_unit = np.array(original_weights / b2.mV)
    except:
        loaded_w_no_unit = np.array(loaded_weights)
        orig_w_no_unit = np.array(original_weights)

    np.testing.assert_allclose(
        loaded_w_no_unit, 
        orig_w_no_unit, 
        rtol=1e-5, 
        err_msg="Los pesos cargados no coinciden con los guardados"
    )
    
    print("✅ Test de Pesos: ÉXITO. Las magnitudes físicas se conservan.")
    
    # Limpieza
    filepath = os.path.join(get_weights_dir(), test_filename)
    if os.path.exists(filepath):
        os.remove(filepath)

# =============================================================================
#  TEST 2: Encoder ECG y Ordenamiento Temporal
# =============================================================================

def test_ecg_encoder_sorting():
    # 1. Configuración
    n_input = 40
    # Inicializamos sin simulación de brian2 para evitar conflictos
    loader = RealECGLoader(n_input=n_input)
    
    fs = 100.0 
    duration = 1.0 
    t = np.linspace(0, duration, int(fs*duration))
    signal = np.sin(2 * np.pi * 5 * t) 
    
    # 2. Ejecutar Encoder
    print("\n[TEST] Ejecutando encoder ECG -> Spikes...")
    indices, times = loader.ecg_to_spikes(signal, fs, duration)
    
    # 3. Conversión de unidades
    try:
        times_sec = np.array(times / b2.second)
    except:
        times_sec = np.array(times)
        
    indices = np.array(indices)
    
    # 4. Aserciones
    assert len(times_sec) > 0, "El encoder no generó ningún spike."
    
    # Check de Ordenamiento (Monotonicidad)
    is_sorted = np.all(np.diff(times_sec) >= 0)
    assert is_sorted, "❌ ERROR CRÍTICO: Spikes desordenados."
    
    assert np.min(times_sec) >= 0, "Tiempos negativos."
    assert np.max(indices) < n_input, "Índices fuera de rango."
    
    print(f"✅ Test de Encoder: ÉXITO. {len(times_sec)} spikes ordenados.")

if __name__ == "__main__":
    test_weight_manager_roundtrip()
    test_ecg_encoder_sorting()