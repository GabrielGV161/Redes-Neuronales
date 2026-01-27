# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 22:34:07 2026

@author: ggv16
"""

# -*- coding: utf-8 -*-
import brian2 as b2
import numpy as np

def compare_recognition(network_dict, pattern_trained, pattern_novel, duration=4*b2.second, pattern_size=40):
    net = network_dict['net']
    input_group = network_dict['input']
    mon_out = network_dict['mon_out']
    mon_v = network_dict['mon_v'] 
    
    # 1. GUARDAR ESTADO INICIAL (¡ESTO FALTABA!) 💾
    # Guardamos la red limpia antes de meterle ningún estímulo
    net.store()
    
    # --- TEST 1: PATRÓN ENTRENADO (SANO) ---
    print("🧪 Test 1: Sano...")
    input_group.set_spikes(pattern_trained[0], pattern_trained[1])
    net.run(duration)
    
    data_trained = {
        't': np.array(mon_v.t/b2.ms),
        'v': np.array(mon_v.v/b2.mV), 
        'spikes_t': np.array(mon_out.t/b2.ms),
        'spikes_i': np.array(mon_out.i)
    }
    
    # 2. RESTAURAR ESTADO LIMPIO 🔄
    net.restore() 
    
    # --- TEST 2: PATRÓN NOVEL (ARRITMIA) ---
    print("🧪 Test 2: Arritmia...")
    input_group.set_spikes(pattern_novel[0], pattern_novel[1])
    net.run(duration)
    
    data_novel = {
        't': np.array(mon_v.t/b2.ms),
        'v': np.array(mon_v.v/b2.mV),
        'spikes_t': np.array(mon_out.t/b2.ms),
        'spikes_i': np.array(mon_out.i)
    }
    
    stats = {}
    return stats, data_trained, data_novel