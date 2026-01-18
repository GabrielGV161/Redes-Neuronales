# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:12:23 2026

@author: ggv16
"""
import brian2 as b2

def build_network(n_input=20, n_output=20, spike_indices=None, spike_times=None, 
                  connectivity_prob=0.8, learning_enabled=True,
                  topology_data=None):  #  Nuevo parámetro
    b2.start_scope()
    b2.defaultclock.dt = 0.1 * b2.ms 
    
    
    
    # Parámetros del modelo LIF y STDP
    params = {
        'tau_m': 10 * b2.ms,        # Constante de tiempo de membrana
        'v_rest': -70 * b2.mV,      # Potencial de reposo
        'v_threshold': -54 * b2.mV, # Umbral de disparo
        'v_reset': -80 * b2.mV,     # Potencial después de disparar
        'tau_refrac': 2 * b2.ms,    # Período refractario
        'tau_pre': 20 * b2.ms,      # Constante de tiempo pre-sinaptica (STDP)
        'tau_post': 20 * b2.ms,     # Constante de tiempo post-sinaptica (STDP)
        'w_max': 8.0 * b2.mV,       # Peso sináptico máximo
        'dA_plus': 0.5 * b2.mV,    # Incremento en LTP (potenciación)
        'dA_minus': 0.6 * b2.mV     # Incremento en LTD (depresión)
    }
    # CAMBIO: Crear con los datos reales desde el inicio
    # Si no hay datos de spikes, usar dummy para evitar errores
    if spike_indices is None or spike_times is None:
        spike_indices = [0]
        spike_times = [0] * b2.ms
    
    input_group = b2.SpikeGeneratorGroup(
        n_input,              # Número de neuronas
        indices=spike_indices, # Qué neuronas disparan
        times=spike_times,     # Cuándo disparan
        name='Input'
    )
    # Ecuación diferencial del modelo LIF
    eqs_lif = '''
    dv/dt = (v_rest - v)/tau_m : volt (unless refractory)
    '''
    # Esta ecuación dice: el voltaje decae exponencialmente hacia v_rest
    # con constante de tiempo tau_m, a menos que esté en período refractario
    
    output_group = b2.NeuronGroup(
        n_output,                       # Número de neuronas de salida
        eqs_lif,                        # Ecuaciones diferenciales
        threshold='v > v_threshold',    # Condición de disparo
        reset='v = v_reset',            # Qué hacer después de disparar
        refractory=params['tau_refrac'], # Período refractario
        method='exact',                 # Método de integración
        namespace=params,               # Parámetros accesibles
        name='Output'
    )
    # Inicializar voltaje en reposo
    output_group.v = params['v_rest']
    
    # SINAPSIS CON STDP
    eqs_stdp = '''
    w : volt                           # Peso sináptico
    dapre/dt = -apre/tau_pre : 1       # Traza pre-sinaptica (decae exponencialmente)
    dapost/dt = -apost/tau_post : 1    # Traza post-sinaptica (decae exponencialmente)
    '''
    # PARTE CRÍTICA: CONDICIONAL PARA STDP
    if learning_enabled:
     # ═══════════════════════════════════════════════
     # MODO APRENDIZAJE: STDP ACTIVO
     # ═══════════════════════════════════════════════
     
     # Qué hacer cuando llega un spike PRE-sináptico (de input):
         on_pre_eq = '''
         v_post += w                                    # 1. Aumentar voltaje de la neurona post
         apre += 1                                      # 2. Incrementar traza pre-sinaptica
         w = clip(w - dA_minus * apost, 0*mV, w_max)    # 3. LTD: si habia traza post, debilitar peso
         '''
         # Explicación línea 3:
             # - Si la neurona POST disparó recientemente (apost > 0)
             # - Entonces este spike PRE llegó DESPUÉS del spike POST
             # - Esto es "mal timing" → debilitamos la sinapsis (LTD)
             # - clip(...) asegura que w esté entre 0 y w_max
     
     # Qué hacer cuando hay un spike POST-sináptico (de output):
         on_post_eq = '''
         apost += 1                                     # 1. Incrementar traza post-sinaptica
         w = clip(w + dA_plus * apre, 0*mV, w_max)      # 2. LTP: si habia traza pre, fortalecer peso
         '''
     # Explicación línea 2:
     # - Si hubo spikes PRE recientemente (apre > 0)
     # - Entonces este spike POST vino DESPUÉS de spikes PRE
     # - Esto es "buen timing" → fortalecemos la sinapsis (LTP)
     
         print(" Modo: APRENDIZAJE ACTIVO (STDP habilitado)")
     
    else:
     # ═══════════════════════════════════════════════
     # MODO RECONOCIMIENTO: PESOS CONGELADOS
     # ═══════════════════════════════════════════════
     
     # Solo transmitir señal, sin modificar pesos
         on_pre_eq = 'v_post += w'  # Solo aumentar voltaje, nada más
         on_post_eq = ''            # No hacer nada cuando output dispara
     
         print(" Modo: RECONOCIMIENTO (Pesos congelados)")
 
 # Crear objeto Synapses con las ecuaciones correspondientes
    # Crear objeto Synapses con las ecuaciones correspondientes
    synapses = b2.Synapses(
       input_group,         # Neuronas pre-sinapticas (fuente)
       output_group,        # Neuronas post-sinapticas (destino)
       model=eqs_stdp,      # Ecuaciones del modelo
       on_pre=on_pre_eq,    #  Ecuación que depende del modo
       on_post=on_post_eq,  #  Ecuación que depende del modo
       method='exact',
       namespace=params,
       name='Synapses'
   )
   #  CONECTAR CON PROBABILIDAD
    if topology_data is not None:
        # CASO A: RECONSTRUCCIÓN EXACTA (Test)
        # Recibimos índices exactos (i, j) y conectamos solo esos.
        inds_i, inds_j = topology_data
        synapses.connect(i=inds_i, j=inds_j)
        print(f" Conectividad reconstruida: {len(synapses)} sinapsis (fija)")
        
    else:
        # CASO B: GENERACIÓN ALEATORIA (Entrenamiento)
        if connectivity_prob < 1.0:
            synapses.connect(p=connectivity_prob)
            print(f" Conectividad aleatoria (p={connectivity_prob}): {len(synapses)} sinapsis")
        else:
            synapses.connect()
            print(f"Conectividad total: {len(synapses)} sinapsis")

    # Inicializar pesos (importante hacerlo antes de cargar los entrenados)
    synapses.w = 3.0 * b2.mV
    
    # MONITORES (para grabar datos durante la simulación)
    spikemon_in = b2.SpikeMonitor(input_group)      # Graba spikes de entrada
    spikemon_out = b2.SpikeMonitor(output_group)    # Graba spikes de salida
    statemon_v = b2.StateMonitor(output_group, 'v', record=True)  # Graba voltaje
    
    # Graba pesos cada 50ms (para no saturar RAM)
    statemon_w = b2.StateMonitor(synapses, 'w', record=True, dt=50*b2.ms)
    
    # CREAR RED Y RETORNAR
    
    # Network() agrupa todos los objetos 
    # DEFINICIÓN EXPLÍCITA (Sin b2.collect)
    net = b2.Network(
        input_group, 
        output_group, 
        synapses, 
        spikemon_in, 
        spikemon_out, 
        statemon_v, 
        statemon_w
    )
    
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
