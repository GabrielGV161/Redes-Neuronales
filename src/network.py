# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:12:23 2026

@author: ggv16
"""
import brian2 as b2

def build_network(n_input=20, n_output=20, spike_indices=None, spike_times=None, 
                  connectivity_prob=0.8, learning_enabled=True):  # ⭐ Nuevo parámetro
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
    statemon_w = b2.StateMonitor(synapses, 'w', record=True, dt=50*b2.ms)
#Graba cada 50 ms en dt para no saturar la RAM

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
