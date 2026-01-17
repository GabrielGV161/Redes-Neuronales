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
    # 1. GENERAR NOMBRE DE ARCHIVO SI NO SE PROPORCIONÓ
    if filename is None:
        # Formato: weights_20260103_143025.pkl (año/mes/día_hora/min/seg)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"weights_{timestamp}.pkl"
    
    
    # 2. CREAR CARPETA PARA GUARDAR (si no existe)
    os.makedirs("saved_weights", exist_ok=True)
    filepath = os.path.join("saved_weights", filename)
    
    # 3. EXTRAER DATOS DE LA RED
    # Obtener pesos como array de NumPy (con unidades de Brian2)
    weights = network_dict['synapses'].w[:]
    
    # Obtener estructura de conectividad
    # pre_indices[i] = neurona de entrada que conecta en sinapsis i
    # post_indices[i] = neurona de salida que conecta en sinapsis i
    pre_indices = network_dict['synapses'].i[:]
    post_indices = network_dict['synapses'].j[:]
    
    # 4. EMPAQUETAR TODO EN UN DICCIONARIO
    data = {
        'weights': weights,           # Pesos sinápticos (con unidades)
        'pre_indices': pre_indices,   # Conectividad pre
        'post_indices': post_indices, # Conectividad post
        'n_synapses': len(network_dict['synapses']),  # Total de sinapsis
        'timestamp': datetime.now().isoformat(),      # Cuándo se guardó
        'metadata': metadata or {}    # Info adicional (vacío si no se pasa)
    }
    
    # 5. GUARDAR EN DISCO (formato pickle)
    # pickle serializa objetos de Python (incluyendo arrays de NumPy)
    with open(filepath, 'wb') as f:  # 'wb' = write binary
        pickle.dump(data, f)
    
    # 6. FEEDBACK AL USUARIO
    print(f"✅ Pesos guardados en: {filepath}")
    print(f"   - {len(weights)} sinapsis")
    print(f"   - Rango: {weights.min():.3f} a {weights.max():.3f}")
    
    return filepath


def load_weights(network_dict, filepath):
    """
    Carga pesos desde un archivo y los aplica a la red.
    
    Args:
        network_dict: Red donde cargar los pesos (debe tener misma estructura)
        filepath: Ruta del archivo .pkl
    
    Returns:
        dict: Metadata del archivo cargado
    
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si la red no es compatible (diferente número de sinapsis)
    
    Ejemplo:
        >>> objs_test = build_network(..., learning_enabled=False)
        >>> load_weights(objs_test, "saved_weights/weights_20260103_143025.pkl")
    """
    # 1. VERIFICAR QUE EL ARCHIVO EXISTE
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ No se encontró: {filepath}")
    
    # 2. CARGAR DATOS DEL ARCHIVO
    with open(filepath, 'rb') as f:  # 'rb' = read binary
        data = pickle.load(f)
    
    # 3. VERIFICAR COMPATIBILIDAD
    # La red actual debe tener el mismo número de sinapsis
    if len(network_dict['synapses']) != data['n_synapses']:
        raise ValueError(
            f"❌ Incompatibilidad: "
            f"red tiene {len(network_dict['synapses'])} sinapsis, "
            f"archivo tiene {data['n_synapses']}"
        )
    
    # 4. APLICAR PESOS A LA RED
    # Sobrescribir los pesos actuales con los cargados
    network_dict['synapses'].w = data['weights']
    
    # 5. FEEDBACK AL USUARIO
    print(f"✅ Pesos cargados desde: {filepath}")
    print(f"   - Guardado el: {data['timestamp']}")
    print(f"   - {data['n_synapses']} sinapsis restauradas")
    
    # Retornar metadata por si es útil
    return data['metadata']


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
    print("\n📊 Cambio en pesos:")
    print(f"   Patrón: {stats['pattern_avg_change']:+.3f} mV "
          f"(final: {stats['pattern_final_avg']:.3f} mV)")
    print(f"   Ruido:  {stats['noise_avg_change']:+.3f} mV "
          f"(final: {stats['noise_final_avg']:.3f} mV)")
    print(f"   Separación: {stats['separation']:.3f} mV")
    
    # Interpretación
    if stats['separation'] > 0.5:
        print("   ✅ Buen aprendizaje (patrón >> ruido)")
    elif stats['separation'] > 0.2:
        print("   ⚠️  Aprendizaje moderado")
    else:
        print("   ❌ Poco o ningún aprendizaje")
    
    return stats