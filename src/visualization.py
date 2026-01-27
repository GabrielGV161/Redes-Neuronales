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
    
def plot_recognition_comparison(data_trained, data_novel, v_threshold_mv=-50.0, title="Diagnóstico SNN"):
    """
    Pinta las trazas de voltaje separando equipos por color.
    
    Args:
        v_threshold_mv (float): Valor del umbral en mV (ej: -50). La línea roja se ajustará a esto.
    """
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Definimos punto medio para separar equipos (asumiendo 10 neuronas)
    n_neurons = data_trained['v'].shape[0] # Filas de la matriz de voltaje
    mid_point = n_neurons // 2
    
    # --- FUNCIÓN AUXILIAR DE PINTADO ---
    def plot_voltage_lines(ax, data, subtitle):
        time = data['t']
        voltages = data['v']
        
        # Iteramos por cada neurona para darle su color
        for i in range(n_neurons):
            if i < mid_point:
                # EQUIPO SANO (A) -> VERDE
                color = 'green'
                label = 'Equipo Sano' if i == 0 else "" # Solo poner label una vez
                zorder = 2 # Poner encima
            else:
                # EQUIPO ARRITMIA (B) -> ROJO
                color = '#D62728' # Rojo bonito
                label = 'Equipo Arritmia' if i == mid_point else ""
                zorder = 1
            
            # Pintamos la línea
            ax.plot(time, voltages[i, :], color=color, alpha=0.8, linewidth=1.2, label=label, zorder=zorder)

        # Configuración del Eje
        ax.set_title(subtitle, fontsize=12)
        ax.set_xlabel("Tiempo (ms)")
        ax.set_ylim(-85, v_threshold_mv + 10) # Ajuste automático del eje Y
        ax.grid(True, alpha=0.3)
        
        # LÍNEA DE UMBRAL DINÁMICA
        ax.axhline(y=v_threshold_mv, color='red', linestyle='--', linewidth=2, label=f'Umbral ({v_threshold_mv} mV)')
        
        # Solo mostrar leyenda en el primer gráfico para no tapar
        if subtitle == "Reacción ante Estímulo SANO":
            ax.legend(loc='lower right', frameon=True, fontsize='small')

    # --- PANEL IZQUIERDO: RESPUESTA A SANO ---
    plot_voltage_lines(axs[0], data_trained, "Reacción ante Estímulo SANO")
    axs[0].set_ylabel("Voltaje de Membrana (mV)")

    # --- PANEL DERECHO: RESPUESTA A ARRITMIA ---
    plot_voltage_lines(axs[1], data_novel, "Reacción ante Estímulo ARRITMIA")

    plt.tight_layout()
    plt.savefig("diagnostico_voltaje_equipos.png")
    plt.show()

def plot_ecg_validation(signal, fs, spike_times, title="Validación ECG Real"):
    """
    Pinta el ECG original y marca dónde la SNN 've' los latidos.
    CORREGIDO: Gestiona unidades de Brian2 para evitar DimensionMismatchError.
    """
    import matplotlib.pyplot as plt
    import brian2 as b2
    import numpy as np
    
    # --- FIX CRÍTICO: Eliminar unidades de Brian2 ---
    # Si spike_times tiene unidades (es un Quantity), lo pasamos a segundos puros (float)
    try:
        times_sec = spike_times / b2.second
    except:
        # Si ya era float (sin unidades), lo dejamos tal cual
        times_sec = spike_times
        
    # Asegurarnos de que es un array de numpy plano
    times_sec = np.array(times_sec)
    # -----------------------------------------------
    
    # Crear eje de tiempos para la señal analógica
    duration = len(signal) / fs
    t_signal = np.linspace(0, duration, len(signal))
    
    plt.figure(figsize=(12, 4))
    plt.plot(t_signal, signal, 'k-', alpha=0.6, label='ECG Real (MIT-BIH)')
    
    # Pintar marcas donde hemos generado spikes
    # Filtramos para pintar solo una línea por latido (clustering visual)
    clean_times = []
    last_t = -1.0 # Float puro
    
    # Ordenamos los tiempos limpios
    for t in np.sort(times_sec):
        # Ahora t y last_t son floats puros, la resta funciona
        if t - last_t > 0.1: # Si ha pasado más de 100ms, asumimos nuevo latido
            clean_times.append(t)
            last_t = t
            
    plt.vlines(clean_times, ymin=np.min(signal), ymax=np.max(signal), 
               colors='r', linestyles='--', linewidth=1.5, label='Latido Detectado (SNN)')
    
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"validation_{title.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"📈 Gráfica de validación guardada: {filename}")
    try:
        plt.show()
    except:
        pass