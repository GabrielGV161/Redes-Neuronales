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
import numpy as np
import pickle
import os
from datetime import datetime
import brian2 as b2



#  AÑADIR AL INICIO DEL ARCHIVO 
def get_weights_dir():
    """
    Obtiene la ruta de la carpeta de pesos, creándola si no existe.
    
    Usa ruta relativa al archivo actual, así funciona en cualquier PC.
    
    Returns:
        str: Ruta absoluta de la carpeta saved_weights/
    """
    # __file__ = ruta de weight_manager.py
    # os.path.dirname(__file__) = carpeta src/
    # os.path.dirname(os.path.dirname(__file__)) = carpeta del proyecto/
    
    # Obtener carpeta raíz del proyecto
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Crear ruta a saved_weights/
    weights_dir = os.path.join(project_root, "saved_weights")
    
    # Crear carpeta si no existe
    os.makedirs(weights_dir, exist_ok=True)
    
    return weights_dir

def save_weights(network_dict, filename=None, metadata=None):
    """
    Guarda los pesos sinápticos en un archivo .pkl
    
    Args:
        network_dict: Diccionario retornado por build_network()
        filename: Nombre del archivo (si None, genera automático con timestamp)
        metadata: Diccionario con información adicional (opcional)
            Ejemplo: {'pattern': 'A', 'duration_ms': 5000, 'description': '...'}
    
    Returns:
        str: Ruta completa del archivo guardado
    
    Ejemplo:
        >>> objs = build_network(...)
        >>> # Entrenar...
        >>> save_weights(objs, "mi_red_entrenada.pkl", {'notas': 'Primera prueba'})
    """
    synapses = network_dict['synapses']
    
    # Creamos la carpeta si no existe
    os.makedirs("saved_weights", exist_ok=True)
    filepath = os.path.join("saved_weights", filename)
    
    data = {
        'weights': np.array(synapses.w),  # Los valores de los pesos
        #  ESTO ES LO NUEVO: Guardamos el mapa de conexiones
        'indices_i': np.array(synapses.i), # Quién envía (pre)
        'indices_j': np.array(synapses.j), # Quién recibe (post)
        'metadata': metadata
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f" Guardado: {filename} ({len(synapses)} sinapsis y su topología)")

def load_topology(filename):
    """
     FUNCIÓN NUEVA:
    Carga SOLO los índices de conexión para poder construir la red igual.
    """
    filepath = os.path.join("saved_weights", filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encuentra: {filepath}")
        
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
    return data['indices_i'], data['indices_j']


def load_weights(network_dict, filename):
    """
    Carga los valores de los pesos en una red YA construida.
    """
    filepath = os.path.join("saved_weights", filename)
    synapses = network_dict['synapses']
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    saved_w = data['weights']
    
    # Verificación de seguridad
    if len(synapses) != len(saved_w):
        # Si esto falla, es que no usamos load_topology al construir
        raise ValueError(
            f"Error de Topología: La red tiene {len(synapses)} sinapsis "
            f"pero el archivo tiene {len(saved_w)}. "
            "¿Usaste 'load_topology' en build_network?"
        )
        
    synapses.w = saved_w * b2.volt  # Restauramos unidades (si se guardaron sin ellas, ajusta esto)
    print(f" Pesos cargados desde {filename}")


def list_saved_weights():
    """
    Lista todos los archivos de pesos guardados.
    
    Returns:
        list: Lista de nombres de archivo disponibles
    
    Ejemplo:
        >>> list_saved_weights()
        ['weights_20260103_143025.pkl', 'pattern_A_trained.pkl']
    """
    weights_dir = get_weights_dir()
    
    if not os.path.exists(weights_dir):
        print(f" No hay carpeta de pesos en: {weights_dir}")
        return []
    
    files = [f for f in os.listdir(weights_dir) if f.endswith('.pkl')]
    
    if not files:
        print(f" Carpeta vacía: {weights_dir}")
    else:
        print(f"\n Pesos guardados en {weights_dir}:")
        for f in files:
            filepath = os.path.join(weights_dir, f)
            size_kb = os.path.getsize(filepath) / 1024
            print(f"   - {f} ({size_kb:.1f} KB)")
    
    return files


def compare_weights(weights_before, weights_after, pattern_synapses_mask):
    """
    Compara pesos antes y después del entrenamiento.
    
    Esta función calcula estadísticas del cambio en los pesos,
    separando las sinapsis del patrón de las del ruido.
    
    Args:
        weights_before: Array de pesos iniciales (antes de entrenar)
        weights_after: Array de pesos finales (después de entrenar)
        pattern_synapses_mask: Array booleano [True, False, True, ...]
            True = sinapsis pertenece al patrón
            False = sinapsis es ruido
    
    Returns:
        dict: Estadísticas del cambio
            - pattern_avg_change: Cambio promedio en pesos del patrón
            - noise_avg_change: Cambio promedio en pesos del ruido
            - separation: Diferencia entre ambos
            - pattern_final_avg: Peso final promedio del patrón
            - noise_final_avg: Peso final promedio del ruido
    
    Ejemplo:
        >>> weights_before = network['synapses'].w[:].copy()
        >>> # Entrenar...
        >>> weights_after = network['synapses'].w[:]
        >>> mask = network['synapses'].i[:] < pattern_size  # True si es patrón
        >>> stats = compare_weights(weights_before, weights_after, mask)
    """
    # 1. CALCULAR CAMBIO (delta)
    # Delta positivo = peso aumentó
    # Delta negativo = peso disminuyó
    delta = weights_after - weights_before
    
    # 2. SEPARAR PATRÓN Y RUIDO USANDO LA MÁSCARA
    # Cambio promedio en sinapsis del patrón
    pattern_change = delta[pattern_synapses_mask].mean()
    
    # Cambio promedio en sinapsis del ruido (~mask invierte la máscara)
    noise_change = delta[~pattern_synapses_mask].mean()
    
    # 3. CALCULAR SEPARACIÓN
    # Separación = qué tan diferente terminaron los pesos del patrón vs ruido
    # Valores altos = buen aprendizaje
    # Valores bajos = no hay diferencia (no aprendió)
    
    # 4. EMPAQUETAR ESTADÍSTICAS
    stats = {
        'pattern_avg_change': float(pattern_change / b2.mV),  # Convertir a float
        'noise_avg_change': float(noise_change / b2.mV),
        'separation': float((pattern_change - noise_change) / b2.mV),
        'pattern_final_avg': float(weights_after[pattern_synapses_mask].mean() / b2.mV),
        'noise_final_avg': float(weights_after[~pattern_synapses_mask].mean() / b2.mV)
    }
    
    # 5. MOSTRAR RESULTADOS
    print("\n  Cambio en pesos:")
    print(f"   Patrón: {stats['pattern_avg_change']:+.3f} mV "
          f"(final: {stats['pattern_final_avg']:.3f} mV)")
    print(f"   Ruido:  {stats['noise_avg_change']:+.3f} mV "
          f"(final: {stats['noise_final_avg']:.3f} mV)")
    print(f"   Separación: {stats['separation']:.3f} mV")
    
    # Interpretación
    if stats['separation'] > 0.5:
        print("    Buen aprendizaje (patrón >> ruido)")
    elif stats['separation'] > 0.2:
        print("     Aprendizaje moderado")
    else:
        print("    Poco o ningún aprendizaje")
    
    return stats