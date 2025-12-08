# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temp``
"""
from brian2 import *

# Parámetros del modelo LIF
tau_m = 10*ms        # Constante de tiempo de membrana
v_rest = -70*mV      # Potencial de reposo
v_threshold = -54*mV # Umbral de disparo
v_reset = -80*mV     # Potencial de reset tras disparo
tau_refrac = 2*ms    # Período refractario

# Ecuaciones del modelo LIF
eqs_lif = '''
dv/dt = (v_rest - v + I_ext)/tau_m : volt (unless refractory)
I_ext : volt
'''

# Crear grupo de neuronas
N = NeuronGroup(100, eqs_lif, 
                threshold='v > v_threshold',
                reset='v = v_reset',
                refractory=tau_refrac,
                method='euler')
N.v = v_rest


