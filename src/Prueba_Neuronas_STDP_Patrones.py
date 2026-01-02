# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 19:30:40 2025

@author: ggv16
"""

# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Qt5Agg')  # Habilitar ventanas interactivas
import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt

def generate_pattern_data(n_input, duration_ms, pattern_size=3, 
                          noise_rate=20*b2.Hz, dt=0.1):
    """
    Genera datos adaptables a cualquier número de neuronas.
    """
    duration_val = duration_ms
    pattern_interval = 50  # Cada 50ms se repite el patrón
    dt_pattern = 5         # Separación entre neuronas del patrón
    
    if pattern_size > n_input:
        pattern_size = n_input
        print(f"Aviso: pattern_size reducido a {n_input}")
    
    indices = []
    times = []
    
    # --- 1. PATRÓN (sin repeticiones extras) ---
    n_repetitions = int(duration_val / pattern_interval)
    
    for i in range(n_repetitions):
        base_time = i * pattern_interval
        for p_idx in range(pattern_size):
            indices.append(p_idx)
            t = base_time + p_idx * dt_pattern  # ⭐ Simplificado
            t_rounded = np.round(t / dt) * dt
            times.append(t_rounded)
    
    # --- 2. RUIDO (igual que antes) ---
    noise_neurons_list = range(pattern_size, n_input)
    noise_rate_val = float(noise_rate / b2.Hz)
    n_noise_spikes_per_neuron = int(noise_rate_val * duration_val / 1000)
    
    for n_idx in noise_neurons_list:
        rand_times = np.random.rand(n_noise_spikes_per_neuron) * duration_val
        rand_times_rounded = np.round(rand_times / dt) * dt
        unique_times = np.unique(rand_times_rounded)
        
        indices.extend([n_idx] * len(unique_times))
        times.extend(unique_times)
    
    # --- 3. ARRAYS ---
    all_indices = np.array(indices, dtype=int)
    all_times = np.array(times)
    
    # --- 4. DEDUPLICAR ---
    spike_pairs = list(zip(all_indices, all_times))
    unique_pairs = list(set(spike_pairs))
    
    if len(unique_pairs) < len(spike_pairs):
        print(f"Aviso: Se eliminaron {len(spike_pairs) - len(unique_pairs)} spikes duplicados")
    
    unique_pairs.sort(key=lambda x: x[1])
    final_indices = np.array([p[0] for p in unique_pairs], dtype=int)
    final_times_vals = np.array([p[1] for p in unique_pairs])
    
    # --- 5. UNIDADES ---
    final_times = final_times_vals * b2.ms
    
    print(f"generate_pattern_data generó:")
    print(f"  - {len(final_indices)} spikes únicos")
    print(f"  - Indices min/max: {final_indices.min()}/{final_indices.max()}")
    print(f"  - Times min/max: {final_times.min()}/{final_times.max()}")
    
    return final_indices, final_times

def build_network(n_input=20, n_output=20, spike_indices=None, spike_times=None, 
                  connectivity_prob=0.8):  # ⭐ Nuevo parámetro
    b2.start_scope()
    b2.defaultclock.dt = 0.1 * b2.ms 
    
    
    
    params = {
        'tau_m': 10 * b2.ms,
        'v_rest': -70 * b2.mV,
        'v_threshold': -54 * b2.mV,
        'v_reset': -80 * b2.mV,
        'tau_refrac': 2 * b2.ms,
        'tau_pre': 20 * b2.ms,
        'tau_post': 20 * b2.ms,
        'w_max': 6.0 * b2.mV,
        'dA_plus': 0.18 * b2.mV,
        'dA_minus': 0.2 * b2.mV
    }

    # CAMBIO: Crear con los datos reales desde el inicio
    if spike_indices is None or spike_times is None:
        spike_indices = [0]
        spike_times = [0] * b2.ms
    
    input_group = b2.SpikeGeneratorGroup(n_input, 
                                         indices=spike_indices, 
                                         times=spike_times, 
                                         name='Input')
   
    eqs_lif = '''
    dv/dt = (v_rest - v)/tau_m : volt (unless refractory)
    '''
    output_group = b2.NeuronGroup(n_output, eqs_lif,
                                  threshold='v > v_threshold',
                                  reset='v = v_reset',
                                  refractory=params['tau_refrac'],
                                  method='exact', 
                                  namespace=params, 
                                  name='Output')
    output_group.v = params['v_rest']

    eqs_stdp = '''
    w : volt
    dapre/dt = -apre/tau_pre : 1 (event-driven)
    dapost/dt = -apost/tau_post : 1 (event-driven)
    '''
    synapses = b2.Synapses(input_group, output_group, model=eqs_stdp,
                           on_pre='''
                           v_post += w
                           apre += 1
                           w = clip(w - dA_minus * apost, 0*mV, w_max) 
                           ''',
                           on_post='''
                           apost += 1
                           w = clip(w + dA_plus * apre, 0*mV, w_max)
                           ''',
                           method='exact',
                           namespace=params,
                           name='Synapses')
    
   # ⭐ CONECTAR CON PROBABILIDAD
    if connectivity_prob < 1.0:
        synapses.connect(p=connectivity_prob)
        print(f"Conectividad sparse: {len(synapses)} de {n_input * n_output} posibles sinapsis")
    else:
        synapses.connect()  # Conexión total
        print(f"Conectividad total: {len(synapses)} sinapsis")
    
    synapses.w = 3 * b2.mV 

    spikemon_in = b2.SpikeMonitor(input_group)
    spikemon_out = b2.SpikeMonitor(output_group)
    statemon_v = b2.StateMonitor(output_group, 'v', record=True)
    statemon_w = b2.StateMonitor(synapses, 'w', record=True, dt=1*b2.ms)

    net = b2.Network(b2.collect())
    return {
        'net': net,
        'input': input_group,
        'synapses': synapses,
        'mon_in': spikemon_in,
        'mon_out': spikemon_out,
        'mon_v': statemon_v,
        'mon_w': statemon_w,
        'n_output': n_output
    }

def run_simulation(network_dict, duration=1000*b2.ms, pattern_size=5, noise_rate=20*b2.Hz):
    """
    Esta versión ya no necesita set_spikes porque los datos 
    se pasaron al crear el network
    """
    n_total_input = network_dict['input'].N 
    
    print(f"Simulando {duration} con {n_total_input} neuronas de entrada.")
    print(f" -> Patrón en neuronas 0 a {pattern_size-1}")
    print(f" -> Ruido en neuronas {pattern_size} a {n_total_input-1}")

    # Ya no necesitamos generar ni inyectar datos aquí
    # porque ya se hicieron en el main
    
    network_dict['net'].run(duration)
    print("Simulación completada.")
    
    
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
             target_output_weights=0,weight_plot_mode="average",
             duration_ms=DURACION/b2.ms)  # ⭐ Solo sinapsis hacia Output 0
    # ⭐ VERIFICACIÓN DE DEBUG
    print(f"Spikes generados: {len(indices)}")
    print(f"Tiempo máximo de spikes: {times[-1]/b2.ms:.1f} ms")
    print(f"Duración de simulación: {DURACION/b2.ms} ms")