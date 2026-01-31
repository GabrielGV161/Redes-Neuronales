import brian2 as b2

def build_network(n_input=90, n_output=10, spike_indices=None, spike_times=None, 
                  connectivity_prob=0.8, learning_enabled=True,
                  topology_data=None):
    """
    Construye la red SNN con arquitectura DUAL.
    Incluye protección para redes vacías (Fase 2 de carga de pesos).
    """

    b2.start_scope()
    b2.defaultclock.dt = 0.1 * b2.ms 
    
    # 1. PARÁMETROS
    params = {
        'v_rest': -70 * b2.mV, 
        'v_reset': -80 * b2.mV,
        'tau_refrac': 5 * b2.ms,
        'w_max': 4.0 * b2.mV,
        'w_inhib': 250 * b2.mV
    }

    # 2. GRUPOS NEURONALES
    if spike_indices is None: spike_indices = [0]; spike_times = [0]*b2.ms
    input_group = b2.SpikeGeneratorGroup(n_input, indices=spike_indices, times=spike_times, name='Input')
    
    eqs_lif = '''
    dv/dt = (v_rest - v) / tau_m : volt (unless refractory)
    tau_m : second
    v_thresh : volt  # <--- NUEVA VARIABLE POR NEURONA
    '''
    output_group = b2.NeuronGroup(n_output, eqs_lif, threshold='v > v_thresh',
                                  reset='v = v_reset', refractory=params['tau_refrac'],
                                  method='exact', namespace=params, name='Output')
    output_group.v = params['v_rest']

    # --- DIVISIÓN DE EQUIPOS ---
    mid = n_output // 2
    team_sano = output_group[:mid]
    team_arritmia = output_group[mid:]
    
    # Personalidades
    team_sano.tau_m = 10 * b2.ms      
    team_sano.v_thresh[:mid] = -55 * b2.mV
    team_arritmia.tau_m = 80 * b2.ms  
    team_arritmia.v_thresh[mid:] = -50 * b2.mV
    # 3. SINAPSIS DE APRENDIZAJE DIVIDIDAS
    eqs_stdp = '''
    w : volt
    dapre/dt = -apre/tau_pre : 1 (clock-driven)
    dapost/dt = -apost/tau_post : 1 (clock-driven)
    '''
    
    # --- A. REGLA FRANCOTIRADOR (Sano) ---
    params_sano = params.copy()
    params_sano['tau_pre'] = 10 * b2.ms   
    params_sano['tau_post'] = 10 * b2.ms
    
    # CAMBIO AQUÍ: Valores mucho más suaves (0.1 en vez de 2.0)
    params_sano['dA_plus'] = 0.40 * b2.mV   # Antes 2.0 mV (Muy brusco)
    params_sano['dA_minus'] = 0.05 * b2.mV # Antes 1.5 mV
    
    synapses_sano = b2.Synapses(input_group, team_sano, model=eqs_stdp,
                                on_pre='v_post += w; apre += 1; w = clip(w - dA_minus * apost, 0*mV, w_max)' if learning_enabled else 'v_post += w',
                                on_post='apost += 1; w = clip(w + dA_plus * apre, 0*mV, w_max)' if learning_enabled else '',
                                method='exact', namespace=params_sano, name='Syn_Sano')

    # --- B. REGLA ACUMULADOR (Arritmia) ---
    params_arr = params.copy()
    params_arr['tau_pre'] = 60 * b2.ms    
    params_arr['tau_post'] = 60 * b2.ms
    
    # CAMBIO AQUÍ: También suavizamos el castigo
    params_arr['dA_plus'] = 2.5 * b2.mV  # Antes 1.0 mV
    params_arr['dA_minus'] = 0.1 * b2.mV  # Antes 2.5 mV (Esto mataba la red)

    synapses_arritmia = b2.Synapses(input_group, team_arritmia, model=eqs_stdp,
                                    on_pre='v_post += w; apre += 1; w = clip(w - dA_minus * apost, 0*mV, w_max)' if learning_enabled else 'v_post += w',
                                    on_post='apost += 1; w = clip(w + dA_plus * apre, 0*mV, w_max)' if learning_enabled else '',
                                    method='exact', namespace=params_arr, name='Syn_Arritmia')

    # Conexiones
    if topology_data is not None:
        synapses_sano.connect()
        synapses_arritmia.connect()
    else:
        # Aquí es donde connect(p=0) no crea nada en Fase 2
        synapses_sano.connect(p=connectivity_prob)
        synapses_arritmia.connect(p=connectivity_prob)
        if connectivity_prob > 0:
            synapses_sano.w = 0.8 * b2.mV
            synapses_arritmia.w = 0.8 * b2.mV
# --- IGUALDAD DE CONDICIONES ---
        # Damos fuerza a AMBOS para que arranquen disparando sí o sí.
        # Antes el Sano tenía poco peso inicial y moría por su Tau corto.
        synapses_sano.w = '2.8 * mV + rand() * 1.0 * mV'      # <--- SUBIDA FUERTE
        synapses_arritmia.w = '2.8 * mV + rand() * 1.0 * mV'  # <--- IGUALDAD

    # 4. INHIBICIÓN LATERAL
    inhib_S_to_A = b2.Synapses(team_sano, team_arritmia, on_pre='v_post -= w_inhib',
                               namespace=params, name='Inhib_S_A')
    inhib_S_to_A.connect()

    inhib_A_to_S = b2.Synapses(team_arritmia, team_sano, on_pre='v_post -= w_inhib',
                               namespace=params, name='Inhib_A_S')
    inhib_A_to_S.connect()

    # 5. OBJETOS Y MONITORES (SECCIÓN CORREGIDA) 🛠️
    spikemon_in = b2.SpikeMonitor(input_group)
    spikemon_out = b2.SpikeMonitor(output_group)
    statemon_v = b2.StateMonitor(output_group, 'v', record=True)
    
    # --- CORRECCIÓN: SOLO CREAR MONITORES SI HAY CABLES ---
    statemon_w_sano = None
    statemon_w_arr = None
    
    monitors_list = [spikemon_in, spikemon_out, statemon_v]
    
    # Verificamos si hay sinapsis antes de poner el micrófono
    if len(synapses_sano) > 0:
        statemon_w_sano = b2.StateMonitor(synapses_sano, 'w', record=[0], dt=100*b2.ms)
        monitors_list.append(statemon_w_sano)
        
    if len(synapses_arritmia) > 0:
        statemon_w_arr = b2.StateMonitor(synapses_arritmia, 'w', record=[0], dt=100*b2.ms)
        monitors_list.append(statemon_w_arr)

    # 6. CONSTRUCCIÓN DE LA RED
    # Añadimos componentes básicos
    net_objects = [input_group, output_group, synapses_sano, synapses_arritmia, 
                   inhib_S_to_A, inhib_A_to_S]
    # Añadimos los monitores (solo los que existen)
    net_objects.extend(monitors_list)

    net = b2.Network(net_objects) nnn

    return {
        'net': net,
        'input': input_group,
        'output': output_group,
        'synapses_sano': synapses_sano,
        'synapses_arr': synapses_arritmia,
        'mon_out': spikemon_out,
        'mon_v': statemon_v,
        'n_output': n_output,
        # Devolvemos None si no se crearon (la simulación sabrá ignorarlos)
        'mon_w_sano': statemon_w_sano, 
        'mon_w_arr': statemon_w_arr
    }