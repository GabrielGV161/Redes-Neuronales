# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:12:23 2026

@author: ggv16
"""
import brian2 as b2

def build_network(n_input=90, n_output=10, spike_indices=None, spike_times=None, 
                  connectivity_prob=0.8, learning_enabled=True,
                  topology_data=None):
    """
    Construye la red SNN para clasificación NO SUPERVISADA de ECG.
    Usa constantes de tiempo (tau_m) variables para distinguir patrones
    sincrónicos (Sanos) de asincrónicos (Arritmias).
    """
    
    b2.start_scope()
    b2.defaultclock.dt = 0.1 * b2.ms 
    
    # =========================================================
    # 1. PARÁMETROS BIOLÓGICOS
    # =========================================================
    params = {
        'v_rest': -70 * b2.mV,
        'v_threshold': -50 * b2.mV, # Umbral accesible
        'v_reset': -80 * b2.mV,
        'tau_refrac': 5 * b2.ms,
        
        # STDP (Aprendizaje)
        'tau_pre': 20 * b2.ms,
        'tau_post': 20 * b2.ms,
        'w_max': 5.0 * b2.mV,
        'dA_plus': 1.0 * b2.mV,
        'dA_minus': 1.5 * b2.mV,
        
        # Inhibición
        'w_inhib': 50.0 * b2.mV  # ¡Inhibición fuerte para la competencia!
    }

    # =========================================================
    # 2. CAPA DE ENTRADA (Retina/Córnea)
    # =========================================================
    if spike_indices is None or spike_times is None:
        spike_indices = [0]; spike_times = [0]*b2.ms

    input_group = b2.SpikeGeneratorGroup(n_input, indices=spike_indices, times=spike_times, name='Input')
    
    # =========================================================
    # 3. CAPA DE SALIDA (Cerebro) - CON TAU VARIABLE
    # =========================================================
    # Fíjate que 'tau_m' ahora se define como variable (: second) 
    # y no como constante en el namespace.
    eqs_lif = '''
    dv/dt = (v_rest - v) / tau_m : volt (unless refractory)
    tau_m : second
    '''
    
    output_group = b2.NeuronGroup(n_output, eqs_lif, threshold='v > v_threshold',
                                  reset='v = v_reset', refractory=params['tau_refrac'],
                                  method='exact', namespace=params, name='Output')
    
    # Inicialización
    output_group.v = params['v_rest']
    
    # --- CONFIGURACIÓN DE PERSONALIDADES (Unsupervised Logic) ---
    if n_output >= 2:
        mid_point = n_output // 2
        
        # EQUIPO A: DETECTORES DE SINCRONÍA (SANO)
        # Memoria muy corta. Solo disparan si los spikes llegan TODOS A LA VEZ.
        output_group.tau_m[:mid_point] = 10 * b2.ms
        
        # EQUIPO B: INTEGRADORES TEMPORALES (ARRITMIA)
        # Memoria larga. Pueden sumar spikes dispersos (Jitter/Ancho).
        output_group.tau_m[mid_point:] = 80 * b2.ms
        
        print(f" 🎭 PERSONALIDADES CONFIGURADAS (NO SUPERVISADO):")
        print(f"    -> Neuronas 0-{mid_point-1}: Tau=10ms (Especialistas en Sincronía/Sano)")
        print(f"    -> Neuronas {mid_point}-{n_output-1}: Tau=80ms (Especialistas en Dispersión/Arritmia)")
    else:
        # Fallback por si pones n_output=1
        output_group.tau_m = 20 * b2.ms

    # =========================================================
    # 4. SINAPSIS DE APRENDIZAJE (Input -> Output)
    # =========================================================
    eqs_stdp = '''
    w : volt
    dapre/dt = -apre/tau_pre : 1 (clock-driven)
    dapost/dt = -apost/tau_post : 1 (clock-driven)
    '''
    if learning_enabled:
        on_pre_eq = '''
        v_post += w * int(not_refractory_post)
        apre += 1
        w = clip(w - dA_minus * apost, 0*mV, w_max)
        '''
        on_post_eq = '''
        apost += 1
        w = clip(w + dA_plus * apre, 0*mV, w_max)
        '''
    else:
        on_pre_eq = 'v_post += w * int(not_refractory_post)'
        on_post_eq = ''

    synapses = b2.Synapses(input_group, output_group, model=eqs_stdp,
                           on_pre=on_pre_eq, on_post=on_post_eq,
                           method='exact', namespace=params, name='Synapses')
    
    if topology_data is not None:
        synapses.connect(i=topology_data[0], j=topology_data[1])
    else:
        synapses.connect(p=connectivity_prob if connectivity_prob < 1.0 else 1.0)
        # Inicialización alta para facilitar el primer disparo
        synapses.w = 3.0 * b2.mV 

    # =========================================================
    # 5. INHIBICIÓN LATERAL (Competencia entre Equipos)
    # =========================================================
    inhib_synapses_A_to_B = None
    inhib_synapses_B_to_A = None

    if n_output >= 2:
        mid_point = n_output // 2
        
        # Slicing de grupos (referencias, no copias)
        team_fast = output_group[:mid_point]   # Sano
        team_slow = output_group[mid_point:]   # Arritmia
        
        # A. Si el Rápido dispara -> Calla al Lento
        # (Prioridad a la sincronía: si es sano, el lento no debe "confundirse")
        inhib_synapses_A_to_B = b2.Synapses(team_fast, team_slow,
                                            on_pre='v_post -= w_inhib',
                                            namespace=params, name='Inhib_Fast_to_Slow')
        inhib_synapses_A_to_B.connect() # Todos contra todos
        
        # B. Si el Lento dispara -> Calla al Rápido
        # (Winner-Take-All: si ya detecté arritmia, tú cállate)
        inhib_synapses_B_to_A = b2.Synapses(team_slow, team_fast,
                                            on_pre='v_post -= w_inhib',
                                            namespace=params, name='Inhib_Slow_to_Fast')
        inhib_synapses_B_to_A.connect() # Todos contra todos
        
        print(" ⚔️ COMPETENCIA ACTIVADA: Inhibición Cruzada entre equipos.")

    # =========================================================
    # 6. MONITORES Y OBJETOS
    # =========================================================
    spikemon_in = b2.SpikeMonitor(input_group)
    spikemon_out = b2.SpikeMonitor(output_group)
    statemon_v = b2.StateMonitor(output_group, 'v', record=True)
    statemon_w = b2.StateMonitor(synapses, 'w', record=True, dt=100*b2.ms) # Record menos frecuente para ahorrar RAM

    # Lista completa de objetos para la red
    net_objects = [input_group, output_group, synapses, spikemon_in, spikemon_out, statemon_v, statemon_w]
    
    if inhib_synapses_A_to_B is not None:
        net_objects.append(inhib_synapses_A_to_B)
        net_objects.append(inhib_synapses_B_to_A)

    net = b2.Network(net_objects)

    return {
        'net': net,
        'input': input_group,
        'output': output_group,
        'synapses': synapses,
        'mon_in': spikemon_in,
        'mon_out': spikemon_out,
        'mon_v': statemon_v,
        'mon_w': statemon_w,
        'n_output': n_output
    }
