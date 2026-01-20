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
    Weight plot mode = ('single','average' or 'heatmap')
    """
    
    mon_in = d['mon_in']
    mon_out = d['mon_out']
    mon_v = d['mon_v']
    mon_w = d['mon_w']
    synapses = d['synapses'] 
    
    if duration_ms is None:
        duration_ms = float(mon_in.t[-1] / b2.ms)
        
    n_output = mon_v.v.shape[0]
    n_synapses = len(synapses)
    
    # Extraemos índices como arrays de Numpy (rápido)
    pre_indices = synapses.i[:]
    post_indices = synapses.j[:]
    
    # --- PREPARACIÓN DE MATRIZ DE PESOS (VECTORIZADA) ---
    # Convertimos TODO a una matriz pura de numpy de una sola vez.
    # Esto evita acceder a la memoria de Brian2 repetidamente.
    # Forma: (n_sinapsis, n_tiempos)
    all_weights_matrix = mon_w.w / b2.mV
    
    # Máscaras booleanas (True/False) para identificar tipos
    is_pattern_synapse = pre_indices < pattern_size
    
    # --- INICIO DE GRÁFICOS ---
    # sharex=True permite que el zoom se aplique a todas las gráficas a la vez
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # 1. Raster Plot - INPUT
    ax1.plot(mon_in.t/b2.ms, mon_in.i, '.k', ms=2, alpha=0.6)
    ax1.axhline(pattern_size - 0.5, color='blue', linestyle='--', alpha=0.3)
    ax1.text(0, pattern_size + 2, 'Ruido', color='blue', fontsize=9)
    ax1.text(0, 1, 'Patrón', color='green', fontsize=9)
    ax1.set_title('Raster Plot (Input)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Neurona ID')
    
    # 2. Raster Plot - OUTPUT
    if len(mon_out.t) > 0:
        ax2.plot(mon_out.t/b2.ms, mon_out.i, '|r', ms=20, mew=2)
        ax2.set_ylim(-0.5, n_output - 0.5)
        n_spikes = len(mon_out.t)
    else:
        n_spikes = 0
    ax2.set_title(f'Output Spikes ({n_spikes} total)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Neurona ID')

    # 3. Voltaje
    n_to_plot = min(max_neurons_voltage, n_output)
    # Aquí usamos un bucle pequeño porque graficamos pocas líneas (5 max)
    for neuron_idx in range(n_to_plot):
        ax3.plot(mon_v.t/b2.ms, mon_v.v[neuron_idx]/b2.mV, alpha=0.7, linewidth=1.5)
    ax3.axhline(-54, color='r', linestyle='--', alpha=0.5, label='Umbral')
    ax3.set_ylabel('Voltaje (mV)')
    ax3.set_title('Voltaje de Membrana', fontsize=10, fontweight='bold')

    # 4. Pesos (OPTIMIZADO)
    
    if weight_plot_mode == 'single':
        # --- MODO SINGLE VECTORIZADO ---
        target = target_output_weights if target_output_weights is not None else 0
        
        # 1. Filtramos solo las sinapsis que conectan con nuestro target
        # Esto crea una máscara booleana instantánea
        mask_target = post_indices == target
        
        # 2. Extraemos los pesos y los tipos (patrón/ruido) usando la máscara
        weights_target = all_weights_matrix[mask_target]
        types_target = is_pattern_synapse[mask_target] # True si es patrón, False si es ruido
        
        # 3. Graficamos
        # Matplotlib necesita un bucle para graficar líneas individuales con colores distintos,
        # pero ya hemos filtrado los datos, así que el bucle es corto.
        time_array = mon_w.t/b2.ms
        
        # Separamos para graficar en bloque (más rápido que ir línea a línea)
        # Transponemos (.T) para que plot entienda que las columnas son series temporales
        if np.any(types_target):
            ax4.plot(time_array, weights_target[types_target].T, 'g-', alpha=0.7, linewidth=1.5)
            
        if np.any(~types_target): # ~ es NOT (lo contrario de True)
            ax4.plot(time_array, weights_target[~types_target].T, 'k-', alpha=0.1, linewidth=0.5)
            
        # Líneas fantasma para la leyenda
        ax4.plot([], [], 'g-', linewidth=2, label='Patrón')
        ax4.plot([], [], 'k-', linewidth=1, label='Ruido')
        ax4.set_title(f'Pesos hacia Output {target}', fontsize=10, fontweight='bold')

    elif weight_plot_mode == 'average':
        # --- MODO AVERAGE VECTORIZADO ---
        if np.any(is_pattern_synapse):
            mean_pattern = np.mean(all_weights_matrix[is_pattern_synapse], axis=0)
            std_pattern = np.std(all_weights_matrix[is_pattern_synapse], axis=0)
            
            ax4.plot(mon_w.t/b2.ms, mean_pattern, 'g-', linewidth=3, label='Patrón (μ)')
            ax4.fill_between(mon_w.t/b2.ms, 
                             mean_pattern - std_pattern,
                             mean_pattern + std_pattern, color='green', alpha=0.2)
            
        # ~is_pattern_synapse invierte la máscara (lo que no es patrón, es ruido)
        if np.any(~is_pattern_synapse):
            mean_noise = np.mean(all_weights_matrix[~is_pattern_synapse], axis=0)
            std_noise = np.std(all_weights_matrix[~is_pattern_synapse], axis=0)
            
            ax4.plot(mon_w.t/b2.ms, mean_noise, 'k-', linewidth=2, label='Ruido (μ)')
            ax4.fill_between(mon_w.t/b2.ms, 
                             mean_noise - std_noise,
                             mean_noise + std_noise, color='gray', alpha=0.2)
            
        ax4.set_title('Evolución Promedio', fontsize=10, fontweight='bold')

    elif weight_plot_mode == 'heatmap':
        # --- MODO HEATMAP VECTORIZADO (SUPER RÁPIDO) ---
        
        # 1. Obtenemos los índices ordenados: primero los que son True (patrón), luego False?
        # argsort ordena de menor a mayor. False(0) < True(1). 
        # Queremos Patrón arriba. Si patrón es índice bajo, usamos np.argsort directo sobre pre_indices.
        # Una forma robusta: Concatenar los índices explícitamente.
        
        idx_pattern = np.where(is_pattern_synapse)[0]
        idx_noise = np.where(~is_pattern_synapse)[0]
        sorted_indices = np.concatenate([idx_pattern, idx_noise])
        
        # 2. Reordenamos la matriz entera usando esos índices (Fancy Indexing)
        weights_sorted = all_weights_matrix[sorted_indices]
        
        im = ax4.imshow(weights_sorted, aspect='auto', cmap='viridis', interpolation='nearest',
                        extent=[0, duration_ms, n_synapses, 0])
        
        ax4.axhline(len(idx_pattern), color='red', linestyle='--', label='División')
        plt.colorbar(im, ax=ax4, label='Peso (mV)')
        ax4.set_title('Heatmap de Pesos', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Sinapsis (Agrupadas)')

    ax4.set_xlabel('Tiempo (ms)')
    if weight_plot_mode != 'heatmap':
        ax4.set_ylabel('Peso (mV)')
        ax4.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas
    print("\n=== DIAGNÓSTICO ===")
    print(f"Spikes de salida: {n_spikes}")
    
def plot_recognition_comparison(res_A, res_B, title="Discriminación: Entrenado vs Novel"):
    """
    Crea una ventana con 4 subgráficas:
    [ Raster Input A ] [ Raster Input B ]
    [ Voltaje Out A  ] [ Voltaje Out B  ]
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # ==========================================
    # COLUMNA IZQUIERDA: PATRÓN ENTRENADO (A)
    # ==========================================
    
    # 1. Raster Input
    ax1 = axes[0, 0]
    ax1.scatter(res_A['input_times'], res_A['input_indices'], s=1, c='green', alpha=0.5)
    ax1.set_title("Estímulo Entrenado (Input)", color='green', fontweight='bold')
    ax1.set_ylabel("ID Neurona Input")
    ax1.set_ylim(-1, 90) # Ajusta esto a tu TOTAL_NEURONAS
    ax1.grid(True, alpha=0.3)

    # 2. Voltaje Output
    ax2 = axes[1, 0]
    times = res_A['time_trace']
    voltages = res_A['voltage_trace']
    
    # Dibujar trazas de las primeras 10 neuronas para no saturar la gráfica
    # Si quieres ver todas, quita el 'min(10, ...)' y pon solo voltages.shape[0]
    for i in range(min(15, voltages.shape[0])):
        ax2.plot(times, voltages[i], color='green', alpha=0.5, linewidth=1)
        
    # Línea de umbral
    ax2.axhline(-45, color='red', linestyle='--', label='Umbral (-45mV)')
    
    ax2.set_title(f"Respuesta: {int(res_A['total_spikes'])} spikes")
    ax2.set_xlabel("Tiempo (ms)")
    ax2.set_ylabel("Voltaje (mV)")
    # Ajustamos el eje Y para ver bien desde reposo (-70) hasta un poco por encima del umbral
    ax2.set_ylim(-85, -20) 
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True, alpha=0.3)

    # ==========================================
    # COLUMNA DERECHA: PATRÓN NOVEL (B)
    # ==========================================
    
    # 3. Raster Input
    ax3 = axes[0, 1]
    ax3.scatter(res_B['input_times'], res_B['input_indices'], s=1, c='gray', alpha=0.5)
    ax3.set_title("Estímulo Novel (Input)", color='gray', fontweight='bold')
    ax3.set_ylim(-1, 90)
    ax3.grid(True, alpha=0.3)

    # 4. Voltaje Output
    ax4 = axes[1, 1]
    times_B = res_B['time_trace']
    voltages_B = res_B['voltage_trace']
    
    for i in range(min(15, voltages_B.shape[0])):
        ax4.plot(times_B, voltages_B[i], color='black', alpha=0.3, linewidth=1)
        
    ax4.axhline(-45, color='red', linestyle='--')
    
    ax4.set_title(f"Respuesta: {int(res_B['total_spikes'])} spikes")
    ax4.set_xlabel("Tiempo (ms)")
    ax4.set_ylim(-85, -20) # Misma escala Y que el A para ser honestos
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()