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
import numpy as np
import pickle
import os
from datetime import datetime
import brian2 as b2

def get_weights_dir():
    """
    Devuelve la ruta absoluta de la carpeta 'saved_weights' dentro del proyecto.
    - Usa __file__ para obtener la ubicación del archivo actual (src/).
    - Sube un nivel para obtener la raíz del proyecto y crea saved_weights/ si no existe.
    Esto evita crear la carpeta en lugares inesperados cuando el script se ejecuta
    desde otra ruta de trabajo.
    Returns:
        str: Ruta absoluta a saved_weights/
    """
    # project_root = carpeta que contiene la carpeta src/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(project_root, "saved_weights")
    os.makedirs(weights_dir, exist_ok=True)  # crear si falta
    return weights_dir

def save_weights(network_dict, filename=None, metadata=None):
    """
    Guarda los pesos y la topología de la red en un archivo .pkl.
    Estrategia:
      - Convertir los pesos a magnitudes en mV (float) antes de guardar.
      - Guardar índices i,j de las sinapsis para poder reconstruir la topología.
      - Añadir metadata opcional (por ejemplo: pattern, duration_ms).
    Args:
      network_dict: Diccionario retornado por build_network() que contiene 'synapses'.
      filename: Nombre del archivo. Si None, se genera con timestamp.
      metadata: Diccionario con información adicional.
    Returns:
      str: Ruta completa del archivo guardado.
    Notas:
      - Guardar magnitudes (floats) evita problemas con pickling de objetos con unidades.
      - Al guardar la unidad explícita ('mV') facilitamos la carga segura luego.
    """
    synapses = network_dict['synapses']
    weights_dir = get_weights_dir()

    # Generar nombre si no se pasó
    if filename is None:
        filename = datetime.now().strftime("weights_%Y%m%d_%H%M%S.pkl")

    filepath = os.path.join(weights_dir, filename)
    
    # Convertimos los pesos a magnitud en mV como array de floats
    # np.array(synapses.w) devuelve los valores, pero pueden perder unidades; 
    # aquí garantizamos la conversión a float en mV.
    weights_mV = (np.array(synapses.w) / b2.mV).astype(float)  # array de floats (sin unidades)
    
    data = {
        'weights_mV': weights_mV,                      # pesos en mV (floats)
        'unit': 'mV',                                  # unidad declarada
        'indices_i': np.array(synapses.i, dtype=int),  # pre-synaptic indices
        'indices_j': np.array(synapses.j, dtype=int),  # post-synaptic indices
        'metadata': metadata,                          # metadata opcional
        'format_version': 1                             # versionado del formato
    }

    # Escribir archivo binario con pickle
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f" Guardado: {filepath} ({len(synapses)} sinapsis)")
    return filepath

def load_topology(filename):
    """
    Carga sólo la topología (indices_i, indices_j) desde un archivo guardado.
    Útil para construir la red con la misma topología antes de asignar pesos.
    Args:
        filename: Nombre del archivo dentro de saved_weights/
    Returns:
        tuple: (indices_i, indices_j) como arrays numpy.
    Raises:
        FileNotFoundError: si no existe el archivo.
    """
    weights_dir = get_weights_dir()
    filepath = os.path.join(weights_dir, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encuentra: {filepath}")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data.get('indices_i'), data.get('indices_j')

def load_weights(network_dict, filename):
    """
    Carga los pesos desde un archivo y los asigna a network_dict['synapses'].
    Este proceso es robusto ante:
      - Archivos con el nuevo formato ('weights_mV' + 'unit').
      - Archivos antiguos que hubieran guardado 'weights' (posible variedad de formatos).
    Comportamientos:
      - Si la topología actual tiene el mismo número de sinapsis que el archivo, asigna directamente.
      - Si la red está vacía (0 sinapsis) y el archivo contiene indices_i/j, reconstruye las conexiones antes de asignar.
      - Si la red tiene más sinapsis que el archivo, asigna los primeros len(saved_w) pesos (parche).
      - En otros casos lanza un ValueError para evitar silent failures.
    Args:
      network_dict: Diccionario que contiene 'synapses' (objetos de Brian2).
      filename: Nombre del archivo dentro de saved_weights/.
    Raises:
      FileNotFoundError, ValueError
    """
    weights_dir = get_weights_dir()
    filepath = os.path.join(weights_dir, filename)
    synapses = network_dict['synapses']

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encuentra: {filepath}")

    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    # -- Compatibilidad y reconstrucción del objeto saved_w con unidades --
    if 'weights_mV' in data:
        # Formato nuevo esperado: guardamos magnitudes en mV
        saved_magnitudes = np.array(data['weights_mV'], dtype=float)
        saved_unit = data.get('unit', 'mV')
        # Si la unidad en el archivo no es mV, alertamos (puedes ampliar conversión)
        if saved_unit != 'mV':
            raise ValueError(f"Unidad inesperada en archivo: {saved_unit}")
        saved_w = saved_magnitudes * b2.mV  # reconstruimos como Quantity de Brian2
    else:
        # Soportar archivos antiguos que tienen 'weights' (posible comportamiento heterogéneo)
        saved_w = data.get('weights')
        if saved_w is None:
            raise ValueError("Archivo de pesos inválido: no contiene 'weights' ni 'weights_mV'.")
        # Si saved_w ya tiene unidades (es una Quantity), dejamos tal cual
        if hasattr(saved_w, 'dim'):
            # saved_w es una magnitud con unidades de Brian2 -> ok
            pass
        else:
            # Asumimos que es array numérico (float) y por compatibilidad tratamos como mV
            saved_w = np.array(saved_w) * b2.mV

    # Extraer topología si está presente
    saved_i = data.get('indices_i', None)
    saved_j = data.get('indices_j', None)

    # -- Asignación robusta --
    len_saved = len(saved_w)
    len_current = len(synapses)

    if len_current == len_saved:
        # Asignación directa
        synapses.w = saved_w
        print("   -> Estructura coincidente: asignación directa de pesos.")
    elif len_current == 0 and saved_i is not None and saved_j is not None:
        # La red aún no tiene conexiones: conectar con la topología guardada
        print("   -> La red está vacía: reconstruyendo conexiones desde la topología guardada.")
        synapses.connect(i=saved_i, j=saved_j)
        synapses.w = saved_w
    elif len_current > len_saved:
        # Parche temporal: asignar a las primeras posiciones
        print("⚠️ ADVERTENCIA: Más sinapsis en la red que en el archivo; asignando parcialmente.")
        synapses.w[:len_saved] = saved_w
    else:
        # No es posible reconciliar topologías (menos sinapsis en archivo que la red actual)
        raise ValueError(f"Error de Topología: Red={len_current}, Archivo={len_saved}")

    # Verificación final (impresa): imprimimos el máximo para confirmar la carga
    print(f"✅ Pesos cargados correctamente. Máx: {np.max(synapses.w)}")

def list_saved_weights():
    """
    Lista los archivos .pkl disponibles en saved_weights/.
    Retorna la lista de nombres de archivo.
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