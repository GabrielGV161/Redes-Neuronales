# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:08:42 2026

@author: ggv16
"""
import brian2 as b2
import numpy as np
from src.data_gen import generate_pattern_data
from src.network import build_network
from src.simulation import run_simulation
from src.visualization import plot_results
from src.weight_manager import save_weights, load_weights, list_saved_weights  #  Añadir list_saved_weights
from src.test_recognition import compare_recognition  #  NUEVO
from src.weight_manager import save_weights, load_weights, load_topology # Importamos la nueva

if __name__ == '__main__':
    TOTAL_NEURONAS_INPUT = 60
    TAMANO_PATRON = 20
    TOTAL_NEURONAS_OUTPUT=20
    DURACION = 5000*b2.ms
    DURACION_TRAIN = 10000*b2.ms
    DURACION_TEST = 10000*b2.ms
    DT = 0.1
    CONECTIVIDAD = 0.8  #  50% de probabilidad
    
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
        learning_enabled=True  #  STDP activo
    )
    
    # Guardar pesos iniciales
    #  GUARDAR (ahora más simple)
    save_weights(objs_train, "pattern_A_trained.pkl", 
            metadata={'pattern': 'A', 'duration_ms': DURACION_TRAIN/b2.ms})

    # Entrenar
    print("\n Entrenando...")
    run_simulation(objs_train, duration=DURACION_TRAIN, pattern_size=TAMANO_PATRON)
    
    # Visualizar
 
   # Visualizar entrenamiento
    plot_results(objs_train, pattern_size=TAMANO_PATRON, 
                max_neurons_voltage=5,
                weight_plot_mode="average",
                duration_ms=DURACION_TRAIN/b2.ms)

    #  VERIFICACIÓN DE DEBUG
    print(f"Spikes generados: {len(indices_A)}")
    print(f"Tiempo máximo de spikes: {times_A[-1]/b2.ms:.1f} ms")
    print(f"Duración de simulación: {DURACION/b2.ms} ms")
    

    #FASE 2: TEST DE RECONOCIMIENTO
    print("\n" + "="*70)
    print("FASE 2: TEST DE RECONOCIMIENTO")
    print("="*70)
    
    # 1. Recuperar topología
    print(" Leyendo estructura de la red entrenada...")
    indices_i, indices_j = load_topology("pattern_A_trained.pkl")

    # 2. Generar Patrón B (Desplazado)
    indices_B, times_B = generate_pattern_data(
        n_input=TOTAL_NEURONAS_INPUT,
        duration_ms=DURACION_TEST/b2.ms,
        pattern_size=TAMANO_PATRON,
        noise_rate=20*b2.Hz,
        dt=DT
    )
    # Desplazamiento espacial para que sea "nuevo"
    indices_B = (indices_B + TAMANO_PATRON) % TOTAL_NEURONAS_INPUT
    
    # En main.py, FASE 2, después de generar indices_B y desplazarlos:

    print("\n--- VERIFICACIÓN DE PATRONES ---")
# Filtramos solo los indices que forman parte del patrón (los primeros pattern_size)
# Nota: asumiendo que generate_pattern pone el patrón primero en el bucle
# Aunque se ordenen temporalmente, podemos ver los únicos.

    neuronas_activas_A = np.unique(indices_A)
    neuronas_activas_B = np.unique(indices_B)
    
    print(f"Neuronas que usa A: {neuronas_activas_A[:10]} ...") 
    print(f"Neuronas que usa B: {neuronas_activas_B[:10]} ...")

# Comprobar intersección
    comunes = np.intersect1d(neuronas_activas_A, neuronas_activas_B)
    print(f"Neuronas compartidas: {len(comunes)} (Debería ser bajo, solo coincidencias de ruido)")
    print("--------------------------------\n")

    # 3. Construir red UNA SOLA VEZ con la topología cargada
    print(" Construyendo red de test...")
    objs_test = build_network(
        n_input=TOTAL_NEURONAS_INPUT, 
        n_output=TOTAL_NEURONAS_OUTPUT,
        spike_indices=[0],    # Dummy, se actualiza en el test
        spike_times=[0]*b2.ms,
        connectivity_prob=CONECTIVIDAD, 
        learning_enabled=False,         #  Test sin aprendizaje
        topology_data=(indices_i, indices_j) #  Topología fija
    )
    
    # 4. Cargar pesos
    load_weights(objs_test, "pattern_A_trained.pkl")
    
    # 5. Comparar
    comparison = compare_recognition(
        objs_test,
        pattern_trained=(indices_A, times_A),
        pattern_novel=(indices_B, times_B),
        duration=DURACION_TEST,
        pattern_size=TAMANO_PATRON
    )
    
    print("\n Experimento completado")