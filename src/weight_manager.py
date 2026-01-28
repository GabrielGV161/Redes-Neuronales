# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 21:57:39 2026

@author: ggv16
"""

"""
Weight Manager: Guardar, cargar y comparar pesos sinápticos

Este módulo permite persistir el estado de la red entrenada
para usarla después en modo reconocimiento.
"""
# -*- coding: utf-8 -*-
"""
Weight Manager (comentado)
Funciones para guardar, cargar y listar pesos sinápticos de la red.
Diseño:
 - Guardamos las magnitudes de los pesos en mV (floats) junto con la topología (i, j).
 - Uso de una carpeta consistente (get_weights_dir) relativa al proyecto.
 - Compatibilidad con archivos antiguos/formatos distintos mediante comprobaciones.
"""
import brian2 as b2
import pickle
import os
import numpy as np

def get_weights_dir():
    # Crea la carpeta saved_weights si no existe
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(root_dir, 'saved_weights')
    os.makedirs(weights_dir, exist_ok=True)
    return weights_dir

def save_weights(network_objects, filename='weights.pkl', metadata=None):
    """
    Guarda los pesos de LOS DOS equipos (Sano y Arritmia) por separado.
    """
    filepath = os.path.join(get_weights_dir(), filename)
    
    # Extraemos los objetos del diccionario devuelto por build_network
    syn_sano = network_objects['synapses_sano']
    syn_arr = network_objects['synapses_arr']
    
    data = {
        'metadata': metadata,
        # Guardamos datos del Equipo Sano
        'sano': {
            'indices_i': np.array(syn_sano.i[:]), # Indices neurona entrada
            'indices_j': np.array(syn_sano.j[:]), # Indices neurona salida
            'w': np.array(syn_sano.w[:])          # Pesos aprendidos
        },
        # Guardamos datos del Equipo Arritmia
        'arritmia': {
            'indices_i': np.array(syn_arr.i[:]),
            'indices_j': np.array(syn_arr.j[:]),
            'w': np.array(syn_arr.w[:])
        }
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"💾 Pesos guardados en: {filepath}")

def load_weights(network_objects, filename='weights.pkl'):
    filepath = os.path.join(get_weights_dir(), filename)
    if not os.path.exists(filepath):
        print(f"❌ Error: No se encontró {filepath}")
        return

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
    syn_sano = network_objects['synapses_sano']
    syn_arr = network_objects['synapses_arr']
    
    # --- ESTRATEGIA SEGURA ---
    # 1. Desactivamos (pero no intentamos escribir en .w todavía)
    syn_sano.active = False 
    syn_arr.active = False
    
    # 2. Conectamos (Esto crea los "cables" físicos)
    # connect(i=..., j=...) sobreescribe o añade. 
    # Como venimos de una red vacía o necesitamos forzar estos índices:
    syn_sano.connect(i=data['sano']['indices_i'], j=data['sano']['indices_j'])
    syn_arr.connect(i=data['arritmia']['indices_i'], j=data['arritmia']['indices_j'])
    
    # 3. AHORA SÍ cargamos los valores (Una vez que existen los cables)
    syn_sano.w = data['sano']['w'] * b2.volt 
    syn_arr.w = data['arritmia']['w'] * b2.volt
    
    # 4. Reactivamos
    syn_sano.active = True
    syn_arr.active = True
    
    print(f"✅ Memoria restaurada. Sano: {len(syn_sano)} con., Arritmia: {len(syn_arr)} con.")

def load_topology(filename='weights.pkl'):
    """
    Función auxiliar para saber qué topología cargar si fuera necesario.
    Devuelve un diccionario con ambas topologías.
    """
    filepath = os.path.join(get_weights_dir(), filename)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
    return {
        'sano': (data['sano']['indices_i'], data['sano']['indices_j']),
        'arritmia': (data['arritmia']['indices_i'], data['arritmia']['indices_j'])
    }