# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:08:42 2026

@author: ggv16
"""
import brian2 as b2
import numpy as np
from src.data_gen import generate_pattern_data,generate_pattern_B_different
from src.network import build_network
from src.simulation import run_simulation
from src.visualization import plot_results, plot_recognition_comparison
from src.weight_manager import save_weights, load_weights, list_saved_weights  #  Añadir list_saved_weights
from src.test_recognition import compare_recognition  #  NUEVO
from src.weight_manager import save_weights, load_weights, load_topology # Importamos la nueva

if __name__ == '__main__':
    TOTAL_NEURONAS_INPUT = 90
    TAMANO_PATRON = 30
    TOTAL_NEURONAS_OUTPUT=20    
    DURACION_TRAIN = 10000*b2.ms
    DURACION_TEST = 5000*b2.ms
    DT = 0.1
    CONECTIVIDAD = 0.8
    
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
    
    # Entrenar
    print("\n Entrenando...")
    run_simulation(objs_train, duration=DURACION_TRAIN, pattern_size=TAMANO_PATRON)
    
    # Guardar pesos iniciales
    #  GUARDAR (ahora más simple)
    save_weights(objs_train, "pattern_A_trained.pkl", 
            metadata={'pattern': 'A', 'duration_ms': DURACION_TRAIN/b2.ms})
    
 
   # Visualizar entrenamiento
    plot_results(objs_train, pattern_size=TAMANO_PATRON, 
                max_neurons_voltage=5,
                weight_plot_mode="average",
                duration_ms=DURACION_TRAIN/b2.ms)

    #  VERIFICACIÓN DE DEBUG
    print(f"Spikes generados: {len(indices_A)}")
    print(f"Tiempo máximo de spikes: {times_A[-1]/b2.ms:.1f} ms")
    print(f"Duración de simulación: {DURACION_TRAIN/b2.ms} ms")
    

# ==========================================
    # FASE 2: TEST DE RECONOCIMIENTO
    # ==========================================
    print("\n" + "="*70)
    print("FASE 2: TEST DE RECONOCIMIENTO")
    print("="*70)

    # 1. Cargar Topología
    indices_i, indices_j = load_topology("pattern_A_trained.pkl")

    # 2. Construir red
    print("🏗️ Construyendo red de test...")
    objs_test = build_network(
        n_input=TOTAL_NEURONAS_INPUT, 
        n_output=TOTAL_NEURONAS_OUTPUT,
        spike_indices=[0], 
        spike_times=[0]*b2.ms,
        learning_enabled=False,
        topology_data=(indices_i, indices_j)
    )
    
    # 3. Cargar y VERIFICAR Pesos
    load_weights(objs_test, "pattern_A_trained.pkl")
    
    # 4. Diagnóstico de pesos
    pesos_actuales = objs_test['synapses'].w[:]  # Esto mantiene las unidades
    mean_w = float(np.mean(pesos_actuales / b2.mV))  # ⭐ Convertir a número
    max_w = float(np.max(pesos_actuales / b2.mV))    # ⭐ Convertir a número
    min_w = float(np.min(pesos_actuales / b2.mV))    # ⭐ Convertir a número

    
    print("\n🔎 DIAGNÓSTICO DE PESOS CARGADOS:")
    print(f"   Peso Promedio: {mean_w:.3f} mV")
    print(f"   Peso Máximo:   {max_w:.3f} mV")
    print(f"   Peso Mínimo:   {min_w:.3f} mV")
    
    if max_w < 0.01:
        print("   ❌ ERROR: Pesos prácticamente cero")
        raise ValueError("Los pesos no se cargaron correctamente")
    elif max_w > 5.5:
        print("   ⚠️  ADVERTENCIA: Muchos pesos saturados en w_max")
    else:
        print("   ✅ Pesos en rango esperado")

    # 5. Generar patrón B (DIFERENTE temporalmente)
    print("\n📊 Generando patrón B (estructura temporal diferente)...")
    indices_B, times_B = generate_pattern_B_different(
        n_input=TOTAL_NEURONAS_INPUT,
        duration_ms=DURACION_TEST/b2.ms,
        pattern_size=TAMANO_PATRON,
        dt=DT
    )
    
    # 6. Regenerar patrón A para el test (misma estructura)
    print("\n📊 Regenerando patrón A para test...")
    indices_A_test, times_A_test = generate_pattern_data(
        n_input=TOTAL_NEURONAS_INPUT,
        duration_ms=DURACION_TEST/b2.ms,
        pattern_size=TAMANO_PATRON,
        noise_rate=0*b2.Hz,  # ⭐ Sin ruido para test limpio
        dt=DT
    )
    # 7. Verificar solapamiento
    print("\n🔍 Verificando separación de patrones...")
    set_A = set(range(TAMANO_PATRON))
    set_B = set(range(30, 30 + TAMANO_PATRON))
    comunes = set_A.intersection(set_B)
    
    print(f"   Patrón A usa neuronas: 0-{TAMANO_PATRON-1}")
    print(f"   Patrón B usa neuronas: 30-{30+TAMANO_PATRON-1}")
    print(f"   Neuronas compartidas: {len(comunes)}")
    
    if len(comunes) > 0:
        print("   ❌ HAY SOLAPAMIENTO")
    else:
        print("   ✅ Patrones completamente separados")

    # 8. Ejecutar test comparativo
    print("\n🧪 Ejecutando comparación de reconocimiento...")
    
    # AHORA DESEMPAQUETAMOS 3 VALORES (Stats, Resultados A, Resultados B)
    stats, res_A, res_B = compare_recognition(
        objs_test,
        pattern_trained=(indices_A_test, times_A_test),
        pattern_novel=(indices_B, times_B),
        duration=DURACION_TEST,
        pattern_size=TAMANO_PATRON
    )
    
    # 9. GRÁFICA FINAL (Nuevo paso)
    print("\n📈 Generando gráfica comparativa de Voltaje...")
    plot_recognition_comparison(res_A, res_B)
    
    print("\n✅ Experimento completado")