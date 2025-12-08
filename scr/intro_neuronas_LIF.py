# -*- coding: utf-8 -*-
"""
Editor de Spyder

"""
#Fase 2
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
#%%
#Fase 3
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

#Fase 4
# Parámetros STDP
tau_stdp_plus = 20*ms   # Constante de tiempo para potenciación
tau_stdp_minus = 20*ms  # Constante de tiempo para depresión
A_plus = 0.01*mV        # Amplitud de potenciación
A_minus = 0.01*mV       # Amplitud de depresión

# Ecuaciones STDP completas
eqs_stdp = '''
dw/dt = 0*mV/ms : volt (clock-driven)
dapre/dt = -apre/tau_stdp_plus : 1 (event-driven)
dapost/dt = -apost/tau_stdp_minus : 1 (event-driven)
'''

synapses_stdp = Synapses(input_neurons, output_neurons,
                         model=eqs_stdp,
                         on_pre='''
                         v_post += w
                         apre += 1
                         w = clip(w + A_plus*apost, 0*mV, 2*mV)
                         ''',
                         on_post='''
                         apost += 1
                         w = clip(w - A_minus*apre, 0*mV, 2*mV)
                         ''',
                         method='euler')

synapses_stdp.connect(p=0.5)
synapses_stdp.w = 'rand() * w_init'
#%%
#Prueba Recon de Ritmo
# Definir patrón de entrada: 3 pulsos con intervalos de 50ms
pattern_times = [0, 50, 100]*ms
pattern_neurons = [0, 5, 10]  # Neuronas específicas que disparan

# Generar estímulo
@network_operation(dt=1*ms)
def apply_pattern(t):
    if int(t/ms) in [0, 50, 100]:
        input_neurons.rates[pattern_neurons] = 100*Hz
    else:
        input_neurons.rates = 0*Hz

# Ejecutar simulación
run(5000*ms)

# Analizar resultados
spike_monitor_input = SpikeMonitor(input_neurons)
spike_monitor_output = SpikeMonitor(output_neurons)
state_monitor_weights = StateMonitor(synapses_stdp, 'w', record=True)
#%%
#Visualizar aprendizaje
import matplotlib.pyplot as plt

# Raster plot de impulsos
figure(figsize=(12, 8))

subplot(3, 1, 1)
plot(spike_monitor_input.t/ms, spike_monitor_input.i, '.k', markersize=3)
xlabel('Tiempo (ms)')
ylabel('Neurona de entrada')
title('Patrón de entrada')

subplot(3, 1, 2)
plot(spike_monitor_output.t/ms, spike_monitor_output.i, '.r', markersize=3)
xlabel('Tiempo (ms)')
ylabel('Neurona de salida')
title('Respuesta de la red')

subplot(3, 1, 3)
# Evolución de pesos sinápticos
for i in range(min(10, len(state_monitor_weights.w))):
    plot(state_monitor_weights.t/ms, state_monitor_weights.w[i]/mV, alpha=0.5)
xlabel('Tiempo (ms)')
ylabel('Peso sináptico (mV)')
title('Evolución de pesos con STDP')

tight_layout()
savefig('results/figures/learning_process.png', dpi=300)


