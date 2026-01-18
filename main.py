# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:08:42 2026

@author: ggv16
"""
import brian2 as b2
from src.data_gen import generate_pattern_data
from src.network import build_network
from src.simulation import run_simulation
from src.visualization import plot_results
from src.weight_manager import save_weights, load_weights  # ⭐ NUEVO
from src.test_recognition import compare_recognition  # ⭐ NUEVO

if __name__ == '__main__':
    TOTAL_NEURONAS_INPUT = 60
    TAMANO_PATRON = 20
    TOTAL_NEURONAS_OUTPUT=20
    DURACION = 5000*b2.ms
    DURACION_TRAIN = 5000*b2.ms
    DURACION_TEST = 2000*b2.ms
    DT = 0.1
    CONECTIVIDAD = 0.8  # ⭐ 50% de probabilidad
    
    #FASE 1: ENTRENAMIENTO
    
    print("\n" + "="*70)
    print("FASE 1: ENTRENAMIENTO")
    print("="*70)
      
    # Generar patrón A (el que vamos a entrenar)
    indices_A, times_A = generate_pattern_data(
        n_input=TOTAL_NEURONAS_INPUT,
        duration_ms=DURACION_TRAIN/b2.ms,
        pattern_size=TAMANO_PATRON,
        noise_rate=20*b2.Hz,
        dt=DT
    )
    
    # Construir red EN MODO APRENDIZAJE
    objs_train = build_network(
        n_input=TOTAL_NEURONAS_INPUT, 
        n_output=TOTAL_NEURONAS_OUTPUT,
        spike_indices=indices_A, 
        spike_times=times_A,
        connectivity_prob=CONECTIVIDAD,
        learning_enabled=True  # ⭐ STDP activo
    )
    
    # Guardar pesos iniciales
    weights_before = objs_train['synapses'].w[:].copy()
    
    # Entrenar
    print("\n🏋️ Entrenando...")
    run_simulation(objs_train, duration=DURACION_TRAIN, pattern_size=TAMANO_PATRON)
    
    # Visualizar
 
   # Visualizar entrenamiento
    plot_results(objs_train, pattern_size=TAMANO_PATRON, 
                max_neurons_voltage=5,
                weight_plot_mode="average",
                duration_ms=DURACION_TRAIN/b2.ms)
    
    # ⭐ VERIFICACIÓN DE DEBUG
    print(f"Spikes generados: {len(indices_A)}")
    print(f"Tiempo máximo de spikes: {times_A[-1]/b2.ms:.1f} ms")
    print(f"Duración de simulación: {DURACION/b2.ms} ms")
    
    #FASE 2: TEST DE RECONOCIMIENTO
    print("\n" + "="*70)
    print("FASE 2: TEST DE RECONOCIMIENTO")
    print("="*70)
    
    # Generar patrón B (diferente, para comparar)
    indices_B, times_B = generate_pattern_data(
        n_input=TOTAL_NEURONAS_INPUT,
        duration_ms=DURACION_TEST/b2.ms,
        pattern_size=TAMANO_PATRON,
        noise_rate=20*b2.Hz,
        dt=DT
    )
    
    # Construir red EN MODO RECONOCIMIENTO (sin aprendizaje)
    objs_test = build_network(
        n_input=TOTAL_NEURONAS_INPUT, 
        n_output=TOTAL_NEURONAS_OUTPUT,
        spike_indices=indices_A,  # Dummy, lo cambiaremos
        spike_times=times_A,
        connectivity_prob=CONECTIVIDAD,
        learning_enabled=False  # ⭐ Pesos congelados
    )
    
    # Cargar pesos entrenados
    load_weights(objs_test, "saved_weights/pattern_A_trained.pkl")
    
    # Comparar respuesta a patrón A vs B
    comparison = compare_recognition(
        objs_test,
        pattern_trained=(indices_A[:len(indices_A)//2], times_A[:len(times_A)//2]),  # Mitad del patrón A
        pattern_novel=(indices_B[:len(indices_B)//2], times_B[:len(times_B)//2]),    # Mitad del patrón B
        duration=DURACION_TEST,
        pattern_size=TAMANO_PATRON
    )
    
    print("\n✅ Experimento completado")