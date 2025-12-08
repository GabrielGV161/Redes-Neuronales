# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 20:45:08 2025

@author: ggv16
"""
# Crear capa de entrada y capa de salida
N_input = 20
N_output = 10

input_neurons = PoissonGroup(N_input, rates=0*Hz)
output_neurons = NeuronGroup(N_output, eqs_lif,
                             threshold='v > v_threshold',
                             reset='v = v_reset',
                             refractory=tau_refrac)

# Parámetros sinápticos
tau_syn = 5*ms
w_init = 0.5*mV

# Ecuaciones sinápticas con STDP
eqs_synapses = '''
dw/dt = 0*mV/ms : volt (clock-driven)
'''

# Crear conexiones con pesos iniciales aleatorios
synapses = Synapses(input_neurons, output_neurons,
                    model=eqs_synapses,
                    on_pre='v_post += w',
                    method='euler')
synapses.connect(p=0.5)  # Conectar con probabilidad 50%
synapses.w = 'rand() * w_init'
