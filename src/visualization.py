# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:14:40 2026

@author: ggv16
"""
import matplotlib
matplotlib.use('Qt5Agg')  # Habilitar ventanas interactivas
import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np

def plot_results(d, pattern_size=3, max_neurons_voltage=5, 
                 target_output_weights=None, 
                 weight_plot_mode='average',
                 duration_ms=None):
    """
    weight_plot_mode: 'single', 'average', o 'heatmap'
    """
    mon_in = d['mon_in']
    mon_out = d['mon_out']
    mon_v = d['mon_v']
    mon_w = d['mon_w']
    synapses = d['synapses']  # ⭐ Necesitamos acceso a las sinapsis
    
    if duration_ms is None:
        duration_ms = float(mon_in.t[-1] / b2.ms)
        
    # Detectar configuración
    n_output = mon_v.v.shape[0]
    n_synapses = len(synapses)
    
    # ⭐ OBTENER MAPEO REAL DE CONEXIONES
    # synapses.i = índices de neuronas pre-sinápticas (input)
    # synapses.j = índices de neuronas post-sinápticas (output)
    pre_indices = synapses.i[:]  # Array con las neuronas input conectadas
    post_indices = synapses.j[:] # Array con las neuronas output conectadas
    
    plt.figure(figsize=(14, 12))
    
   
    # 1. Raster Plot - INPUT
    plt.subplot(4, 1, 1)  # <--- IMPORTANTE: Índice 1 (arriba del todo)
    
    # Graficamos los datos de entrada (mon_in), no los de salida
    plt.plot(mon_in.t/b2.ms, mon_in.i, '.k', ms=2, alpha=0.6) 
    
    # Línea divisoria visual entre patrón y ruido
    plt.axhline(pattern_size - 0.5, color='blue', linestyle='--', alpha=0.3)
    plt.text(0, pattern_size + 2, 'Ruido', color='blue', fontsize=8)
    plt.text(0, 1, 'Patrón', color='green', fontsize=8)
    
    plt.title('Raster Plot (Input)', fontsize=12, fontweight='bold')
    plt.ylabel('Neurona ID')
    plt.xlim(0, duration_ms)
    
    
    # 2. Raster Plot - OUTPUT
    plt.subplot(4, 1, 2)
    
    # Verificamos si hay datos antes de graficar
    if len(mon_out.t) > 0:
        plt.plot(mon_out.t/b2.ms, mon_out.i, '|r', ms=20, mew=2)
        plt.ylim(-0.5, n_output - 0.5)
        n_spikes = len(mon_out.t)
    else:
        n_spikes = 0
        
    plt.title(f'Output Spikes ({n_output} neuronas, {n_spikes} spikes totales)', 
              fontsize=12, fontweight='bold')
    plt.ylabel('Neurona ID')
    plt.xlim(0, duration_ms)  # ⭐ Ajustar automáticamente
    # 3. Voltaje
    plt.subplot(4, 1, 3)
    n_to_plot = min(max_neurons_voltage, n_output)
    for neuron_idx in range(n_to_plot):
        plt.plot(mon_v.t/b2.ms, mon_v.v[neuron_idx]/b2.mV, 
                alpha=0.7, linewidth=1.5)
    plt.axhline(-54, color='r', linestyle='--', linewidth=2, alpha=0.5, label='Umbral')
    
    v_min = np.min(mon_v.v[:] / b2.mV)
    v_max = np.max(mon_v.v[:] / b2.mV)
    plt.title(f'Voltaje de Membrana ({n_to_plot}/{n_output} neuronas, rango: {v_min:.1f} a {v_max:.1f} mV)', 
              fontsize=12, fontweight='bold')
    plt.ylabel('Voltaje (mV)')
    plt.xlabel('Tiempo (ms)')
    plt.xlim(0, duration_ms)  # ⭐ Ajustar automáticamente
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    
    # 4. Evolución de Pesos (⭐ ADAPTADO PARA SPARSE)
    plt.subplot(4, 1, 4)
    
    if weight_plot_mode == 'single':
        # OPCIÓN A: Una neurona output
        target = target_output_weights if target_output_weights is not None else 0
        pattern_count = 0
        noise_count = 0
        
        for syn_idx in range(n_synapses):
            pre_neuron = pre_indices[syn_idx]
            post_neuron = post_indices[syn_idx]
            
            if post_neuron != target:
                continue
            
            if pre_neuron < pattern_size:
                pattern_count += 1
                plt.plot(mon_w.t/b2.ms, mon_w.w[syn_idx]/b2.mV, 
                        'g-', linewidth=2, alpha=0.7)
            else:
                noise_count += 1
                plt.plot(mon_w.t/b2.ms, mon_w.w[syn_idx]/b2.mV, 
                        'k-', linewidth=0.5, alpha=0.3)
        
        plt.plot([], [], 'g-', linewidth=2, label=f'Patrón ({pattern_count})')
        plt.plot([], [], 'k-', linewidth=1, label=f'Ruido ({noise_count})')
        plt.title(f'Pesos hacia Output {target}', fontsize=12, fontweight='bold')
    
    elif weight_plot_mode == 'average':
        # OPCIÓN B: Promedios
        pattern_weights = []
        noise_weights = []
        
        for syn_idx in range(n_synapses):
            pre_neuron = pre_indices[syn_idx]
            
            if pre_neuron < pattern_size:
                pattern_weights.append(mon_w.w[syn_idx])
            else:
                noise_weights.append(mon_w.w[syn_idx])
        
        if len(pattern_weights) > 0:
            pattern_weights = np.array(pattern_weights)
            mean_pattern = np.mean(pattern_weights, axis=0)
            std_pattern = np.std(pattern_weights, axis=0)
            
            plt.plot(mon_w.t/b2.ms, mean_pattern/b2.mV, 
                     'g-', linewidth=3, label=f'Patrón (μ de {len(pattern_weights)})')
            plt.fill_between(mon_w.t/b2.ms, 
                             (mean_pattern - std_pattern)/b2.mV,
                             (mean_pattern + std_pattern)/b2.mV,
                             color='green', alpha=0.2)
        
        if len(noise_weights) > 0:
            noise_weights = np.array(noise_weights)
            mean_noise = np.mean(noise_weights, axis=0)
            std_noise = np.std(noise_weights, axis=0)
            
            plt.plot(mon_w.t/b2.ms, mean_noise/b2.mV, 
                     'k-', linewidth=2, label=f'Ruido (μ de {len(noise_weights)})')
            plt.fill_between(mon_w.t/b2.ms, 
                             (mean_noise - std_noise)/b2.mV,
                             (mean_noise + std_noise)/b2.mV,
                             color='gray', alpha=0.2)
        
        plt.title('Evolución de Pesos (promedios ± σ)', fontsize=12, fontweight='bold')
    
    elif weight_plot_mode == 'heatmap':
        # OPCIÓN C: Heatmap
        n_timepoints = len(mon_w.t)
        weight_matrix = np.zeros((n_synapses, n_timepoints))
        
        for syn_idx in range(n_synapses):
            weight_matrix[syn_idx, :] = mon_w.w[syn_idx] / b2.mV
        
        pattern_indices = [i for i in range(n_synapses) if pre_indices[i] < pattern_size]
        noise_indices = [i for i in range(n_synapses) if pre_indices[i] >= pattern_size]
        sorted_indices = pattern_indices + noise_indices
        weight_matrix_sorted = weight_matrix[sorted_indices, :]
        
        im = plt.imshow(weight_matrix_sorted, aspect='auto', 
                        cmap='viridis', interpolation='nearest',
                        extent=[0, mon_w.t[-1]/b2.ms, n_synapses, 0])
        
        plt.axhline(len(pattern_indices), color='red', 
                    linestyle='--', linewidth=2, label='División')
        
        plt.colorbar(im, label='Peso (mV)')
        plt.title(f'Heatmap de Pesos ({len(pattern_indices)} patrón, {len(noise_indices)} ruido)', 
                  fontsize=12, fontweight='bold')
        plt.ylabel('Sinapsis (ordenadas)')
    
    plt.legend(loc="upper left")
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('Peso (mV)' if weight_plot_mode != 'heatmap' else 'Sinapsis')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    
    
    
    # Estadísticas
    print("\n=== DIAGNÓSTICO ===")
    print(f"Spikes de salida: {n_spikes}")
    print(f"Voltaje rango: {v_min:.2f} a {v_max:.2f} mV")
    if v_max >= -54:
        print("✅ Las neuronas SÍ alcanzan el umbral")
    else:
        print("⚠️ Las neuronas NO alcanzan el umbral")