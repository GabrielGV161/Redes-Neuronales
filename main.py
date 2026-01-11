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

if __name__ == '__main__':
    TOTAL_NEURONAS_INPUT = 60
    TAMANO_PATRON = 20
    TOTAL_NEURONAS_OUTPUT=20
    DURACION = 5000*b2.ms
    DT = 0.1
    CONECTIVIDAD = 0.8  # ⭐ 50% de probabilidad
    
    # Generar datos
    print("Generando datos...")
    indices, times = generate_pattern_data(
        n_input=TOTAL_NEURONAS_INPUT,
        duration_ms=DURACION/b2.ms,
        pattern_size=TAMANO_PATRON,
        noise_rate=20*b2.Hz,
        dt=DT
    )
    
    # Construir red con conectividad sparse
    print("Construyendo red...")
    objs = build_network(
        n_input=TOTAL_NEURONAS_INPUT, 
        n_output=TOTAL_NEURONAS_OUTPUT,
        spike_indices=indices, 
        spike_times=times,
        connectivity_prob=CONECTIVIDAD  # ⭐ Pasar probabilidad
    )
    
    # Correr
    print("Ejecutando simulación...")
    run_simulation(objs, duration=DURACION, pattern_size=TAMANO_PATRON)
    
    # Visualizar
 
    # En el main, cambia la llamada a plot_results:
    plot_results(objs, pattern_size=TAMANO_PATRON, 
             max_neurons_voltage=5,
             target_output_weights=0,weight_plot_mode="heatmap",
             duration_ms=DURACION/b2.ms)  # ⭐ Solo sinapsis hacia Output 0
    # ⭐ VERIFICACIÓN DE DEBUG
    print(f"Spikes generados: {len(indices)}")
    print(f"Tiempo máximo de spikes: {times[-1]/b2.ms:.1f} ms")
    print(f"Duración de simulación: {DURACION/b2.ms} ms")