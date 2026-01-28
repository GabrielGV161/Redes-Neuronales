# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:08:42 2026
@author: ggv16
"""
import brian2 as b2
import numpy as np

class SimResult:
    """
    Clase auxiliar para empaquetar resultados de simulación.
    Convierte las 'vistas' de Brian2 a arrays reales de Numpy.
    """
    def __init__(self, t, v):
        self.t = np.array(t)
        self.v = np.array(v)

def ensure_seconds(times):
    if not hasattr(times, 'dim'): 
        return times * b2.second
    return times

def compare_recognition(params, pattern_trained, pattern_novel, 
                       topology_save_name='pattern_ecg_trained.pkl',
                       input_gain=1.0):
    
    # --- 1. CONFIGURACIÓN ESTRUCTURAL ---
    b2.start_scope()
    b2.defaultclock.dt = 0.1 * b2.ms
    
    # Recuperamos parámetros
    N_input = params.get('n_input', 90)
    N_output = params.get('n_output', 10)
    tau_sano = params.get('tau_sano', 20*b2.ms) 
    tau_arritmia = params.get('tau_arritmia', 80*b2.ms)
    w_inhib = params.get('w_inhib', 5*b2.mV)
    v_thresh_sano = params.get('v_thresh_sano', -53*b2.mV)     # Default si no existe
    v_thresh_arr = params.get('v_thresh_arr', -50*b2.mV)       # Default si no existe
    # Grupos neuronales
    input_group = b2.SpikeGeneratorGroup(N_input, [], []*b2.ms)
    
    # En src/test_recognition.py

    # --- CÓDIGO LIMPIO SIN COMENTARIOS DENTRO DE LAS COMILLAS ---
    eqs_lif = '''
    dv/dt = (v_rest - v) / tau_m : volt (unless refractory)
    tau_m : second
    v_rest : volt
    v_thresh : volt
    v_reset : volt
    '''
    
    output_group = b2.NeuronGroup(N_output, eqs_lif,
                                  threshold='v > v_thresh', # <--- USA LA VARIABLE
                                  reset='v = v_reset',
                                  refractory=2*b2.ms,
                                  method='exact')
    
    # Física de la neurona
    output_group.v_rest = -70*b2.mV
    output_group.v_reset = -80*b2.mV 
       
    # Personalidades
    n_sano = N_output // 2
    output_group.tau_m[:n_sano] = params.get('tau_sano', 3*b2.ms)
    output_group.tau_m[n_sano:] = params.get('tau_arritmia', 100*b2.ms)
    # --- ASIGNACIÓN DE UMBRALES DIFERENTES ---
    output_group.v_thresh[:n_sano] = v_thresh_sano  # Equipo Verde
    output_group.v_thresh[n_sano:] = v_thresh_arr   # Equipo Rojo
    # Recuperamos parámetros (Asegúrate de definirlos en el main luego)
    
    # =========================================================
    # --- 2. CARGA DE PESOS (SECCIÓN MODIFICADA) ---
    # =========================================================
    from src.weight_manager import load_weights # Usamos tu nuevo gestor
    
    # CAMBIO 1: Creamos DOS grupos de sinapsis en lugar de uno.
    # syn_sano: Conecta Entrada -> Primera mitad (Equipo Verde)
    # syn_arr:  Conecta Entrada -> Segunda mitad (Equipo Rojo)
    
    syn_sano = b2.Synapses(input_group, output_group[:n_sano], 
                           'w : volt', on_pre='v += w', name='synapses_sano')
                           
    syn_arr = b2.Synapses(input_group, output_group[n_sano:], 
                          'w : volt', on_pre='v += w', name='synapses_arr')

    # CAMBIO 2: Preparamos el diccionario para que load_weights sepa dónde poner los datos
    # Las claves ('synapses_sano', 'synapses_arr') deben coincidir con weight_manager.py
    network_dict = {
        'synapses_sano': syn_sano,
        'synapses_arr': syn_arr
    }
    
    # CAMBIO 3: load_weights ahora se encarga de todo (conectar y asignar pesos)
    # Ya no hace falta abrir el pickle manualmente aquí.
    load_weights(network_dict, topology_save_name)
    
    # === AQUI ES EL MOMENTO ===
    # Retrasamos al equipo Rojo 3ms. 
    # Esto da tiempo al Verde a disparar e inhibir antes de que el Rojo reaccione.

    # ==========================
    # APLICAR INPUT GAIN (Opcional, aplica a ambos)
    if input_gain != 1.0:
        syn_sano.w *= input_gain
        syn_arr.w *= input_gain
    
    # =========================================================
    # Fin de cambios críticos en estructura
    # =========================================================

    # Inhibición Lateral (Igual que antes)
    params_inhib = {'w_inhib': w_inhib}
    inhib_S_to_A = b2.Synapses(output_group[:n_sano], output_group[n_sano:],
                               'w_in : volt', on_pre='v -= w_in', namespace=params_inhib)
    inhib_S_to_A.connect()
    
    inhib_A_to_S = b2.Synapses(output_group[n_sano:], output_group[:n_sano],
                               'w_in : volt', on_pre='v -= w_in', namespace=params_inhib)
    inhib_A_to_S.connect()
    
    # Monitores
    spikemon = b2.SpikeMonitor(output_group)
    statemon = b2.StateMonitor(output_group, 'v', record=True)
    
    # CAMBIO 4: Añadimos las dos sinapsis nuevas a la red
    # Quitamos 'synapses' y ponemos 'syn_sano, syn_arr'
    net = b2.Network(input_group, output_group, syn_sano, syn_arr, 
                     inhib_S_to_A, inhib_A_to_S, spikemon, statemon)
    
    net.store() 
    
    # =========================================================
    # TEST 1: PATRÓN ENTRENADO (SANO)
    # =========================================================
    times_trained = ensure_seconds(pattern_trained[1])
    input_group.set_spikes(pattern_trained[0], times_trained)
    
    if len(times_trained) > 0:
        duration_trained = times_trained[-1] + 0.2*b2.second 
    else:
        duration_trained = 200*b2.ms
        
    net.run(duration_trained)
    
    res_healthy = SimResult(statemon.t, statemon.v)
    
    # =========================================================
    # TEST 2: PATRÓN NOVEDOSO (ARRITMIA)
    # =========================================================
    net.restore() 
    
    times_novel = ensure_seconds(pattern_novel[1])
    input_group.set_spikes(pattern_novel[0], times_novel)
    
    if len(times_novel) > 0:
        duration_novel = times_novel[-1] + 0.2*b2.second
    else:
        duration_novel = 1*b2.second
    
    net.run(duration_novel)
    
    res_arrhythmia = SimResult(statemon.t, statemon.v)
    
    # Stats (Mantenemos tu lógica original para el return, aunque te recomiendo usar el print_report)
    spikes_sano = np.sum(spikemon.count[:n_sano])
    spikes_arritmia = np.sum(spikemon.count[n_sano:])
    
    stats = {
        'total_spikes': spikemon.num_spikes,
        'sano_activity': spikes_sano,
        'arritmia_activity': spikes_arritmia
    }
    
    return stats, res_healthy, res_arrhythmia


def print_diagnosis_report(res_healthy, res_arrhythmia, threshold_sano, threshold_arr):
    import numpy as np
    
    print("\n" + "="*60)
    print("      📋 INFORME CLÍNICO FINAL (CONTEO DE EVENTOS)       ")
    print("="*60)

    # Función inteligente que agrupa cruces en "Eventos"
    def contar_eventos_reales(trazas_voltaje, umbral_mv, periodo_refractario_ms=150):
        umbral_val = umbral_mv / 1000.0
        dt = 0.1 # ms (paso de tiempo de la simulación)
        pasos_refractarios = int(periodo_refractario_ms / dt)
        
        total_eventos = 0
        
        # Iteramos por cada neurona del equipo (normalmente 5)
        for i in range(trazas_voltaje.shape[0]):
            v = trazas_voltaje[i]
            ultimo_disparo = -pasos_refractarios
            
            for t in range(1, len(v)):
                # Detectar cruce hacia arriba
                if v[t-1] < umbral_val and v[t] >= umbral_val:
                    # Chequear si ha pasado suficiente tiempo desde el último evento
                    if (t - ultimo_disparo) > pasos_refractarios:
                        total_eventos += 1
                        ultimo_disparo = t
                        
        return total_eventos

    # --- ANÁLISIS ---
    
    # CASO 1: SANO
    # Verde cruza -20mV. Rojo cruza -30mV.
    eventos_verde_1 = contar_eventos_reales(res_healthy.v[:5], threshold_sano)
    eventos_rojo_1  = contar_eventos_reales(res_healthy.v[5:], threshold_arr)

    print(f"\n--- 🏥 CASO 1: PACIENTE SANO ---")
    print(f"   🟢 Eventos Verde (Umbral {threshold_sano} mV): {eventos_verde_1}")
    print(f"   🔴 Eventos Rojo  (Umbral {threshold_arr} mV): {eventos_rojo_1}")

    # LÓGICA DE DECISIÓN CORREGIDA:
    # En un paciente SANO, el Verde debe detectar TODOS los latidos.
    # El Rojo detectará algunos por error, pero MENOS o IGUAL que el Verde.
    # Si Verde >= Rojo -> GANA VERDE (Porque tiene prioridad jerárquica).
    
    if eventos_verde_1 >= eventos_rojo_1 and eventos_verde_1 > 0:
        print("   ✅ DIAGNÓSTICO: PACIENTE SANO")
    elif eventos_rojo_1 > eventos_verde_1:
        print("   ❌ DIAGNÓSTICO: ARRITMIA (Falso Positivo)")
    else:
        print("   ⚠️ INDETERMINADO")

    # CASO 2: ARRITMIA
    eventos_verde_2 = contar_eventos_reales(res_arrhythmia.v[:5], threshold_sano)
    eventos_rojo_2  = contar_eventos_reales(res_arrhythmia.v[5:], threshold_arr)

    print(f"\n--- 💔 CASO 2: PACIENTE ARRITMIA ---")
    print(f"   🟢 Eventos Verde (Umbral {threshold_sano} mV): {eventos_verde_2}")
    print(f"   🔴 Eventos Rojo  (Umbral {threshold_arr} mV): {eventos_rojo_2}")
    
    if eventos_rojo_2 > eventos_verde_2:
        print("   ✅ DIAGNÓSTICO: ARRITMIA")
    elif eventos_verde_2 >= eventos_rojo_2:
        print("   ❌ DIAGNÓSTICO: SANO (Falso Negativo)")
    
    print("\n" + "="*60)