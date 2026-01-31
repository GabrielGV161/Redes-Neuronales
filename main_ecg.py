# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:08:42 2026
@author: ggv16
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:08:42 2026
@author: ggv16
"""
import brian2 as b2
import numpy as np
import gc

# --- MÓDULOS PROPIOS ---
from src.network import build_network
from src.simulation import run_simulation  # <--- USAMOS TU NOMBRE REAL
from src.weight_manager import save_weights, load_weights, load_topology
# En la parte superior de main_ecg.py
from src.test_recognition import compare_recognition, print_diagnosis_report
from src.visualization import plot_recognition_comparison

# Generadores
from src.ecg_gen import generate_ecg_data
from src.real_ecg_loader import RealECGLoader

if __name__ == '__main__':
    # ==========================================
    # ⚙️ CONFIGURACIÓN GLOBAL
    # ==========================================
    TOTAL_NEURONAS_INPUT = 90
    TAMANO_PATRON = 40          
    TOTAL_NEURONAS_OUTPUT = 10   
    
    NUM_EPOCHS = 30             
    DURACION_TRAIN = 200*b2.ms  
    DURACION_TEST = 4*b2.second 
    
    WEIGHTS_FILE = "pattern_ecg_trained.pkl"

    # ==========================================
    # FASE 1: ENTRENAMIENTO DINÁMICO (Simulador)
    # ==========================================
    print(f"\n[{'='*20} FASE 1: ENTRENAMIENTO DINÁMICO {'='*20}]")
    print(f"🏋️‍♂️ Entrenando 'Cerebro' con datos sintéticos ({NUM_EPOCHS} épocas)...")

    # 1. Construir red (STDP Activado)
    objs_train = build_network(
        n_input=TOTAL_NEURONAS_INPUT,
        n_output=TOTAL_NEURONAS_OUTPUT,
        spike_indices=[0],      
        spike_times=[0]*b2.ms,  
        learning_enabled=True,
        connectivity_prob=0.7
    )

    for epoch in range(NUM_EPOCHS):
        current_bpm = np.random.normal(75, 5) 
        
        # --- EL "DJ" ALTERNA LOS DISCOS ---
        # Pares = Sano, Impares = Arritmia
        if epoch % 2 == 0:
            modo_actual = 'healthy'
            print(f"   🔄 Época {epoch+1} | Enseñando: ❤️ SANO")
        else:
            modo_actual = 'arrhythmia'
            print(f"   🔄 Época {epoch+1} | Enseñando: 💔 ARRITMIA")

# ==================================================================
        # 🧠 GATE DE INHIBICIÓN SIMÉTRICO (PROTECCIÓN TOTAL)
        # ==================================================================
        # Recuperamos AMBOS cables de inhibición
        if 'inhib_S_A' in objs_train and 'inhib_A_S' in objs_train:
            gate_S_to_A = objs_train['inhib_S_A'] # Sano pega a Arritmia
            gate_A_to_S = objs_train['inhib_A_S'] # Arritmia pega a Sano
            
            if modo_actual == 'arrhythmia':
                # TURNO DEL ROJO:
                # Desactivamos el ataque del Sano. Activamos el ataque del Rojo (defensa).
                gate_S_to_A.active = False
                gate_A_to_S.active = True 
                # print("      🛡️ Protegiendo al Rojo (S->A apagado)")
                
            else: # modo_actual == 'healthy'
                # TURNO DEL VERDE:
                # Desactivamos el ataque del Rojo. Activamos el ataque del Sano.
                # Esto es CRÍTICO ahora que el Rojo es fuerte, para que no mate al Verde.
                gate_S_to_A.active = True
                gate_A_to_S.active = False
                # print("      🛡️ Protegiendo al Verde (A->S apagado)")
        # ==================================================================

        # Diagnóstico de pesos (Estado actual antes de simular)
        w_sano = objs_train['synapses_sano'].w
        w_arr = objs_train['synapses_arr'].w
        
        print(f"   📊 PESOS MEDIA - Sano: {np.mean(w_sano/b2.mV):.4f} mV | Arritmia: {np.mean(w_arr/b2.mV):.4f} mV")
        print(f"   📊 PESOS MAX   - Sano: {np.max(w_sano/b2.mV):.4f} mV | Arritmia: {np.max(w_arr/b2.mV):.4f} mV")
        
        # Chequeo de disparos (OJO: Esto cuenta el acumulado total, para depurar es mejor mirar diferencias)
        disparos_sano = np.sum(objs_train['mon_out'].count[:5])
        if epoch > 0 and disparos_sano == 0: # Solo alertar si llevamos ya alguna época
             print("   ⚠️ ALERTA: El equipo sano sigue sin disparar.")

        print("-" * 30)
            
        # Generamos con el modo elegido
        indices_train, times_train = generate_ecg_data(
            n_input=TOTAL_NEURONAS_INPUT,
            duration_ms=DURACION_TRAIN/b2.ms,
            bpm=current_bpm,
            mode=modo_actual,
            noise_level=0.1,             # Un poco de ruido está bien
            # --- PARÁMETROS DE FRANCOTIRADOR ---
            jitter_healthy=0.005,    # 2ms: Obligamos a aprender patrones MUY precisos
            jitter_arrhythmia=0.020
        )
        
        # C. Feedback
        print(f"   🔄 Época {epoch+1}/{NUM_EPOCHS} | Ritmo: {current_bpm:.1f} BPM | Spikes: {len(indices_train)}")
        
        # D. CORRECCIÓN TEMPORAL (CRÍTICO PARA QUE FUNCIONE) ⏱️
        current_sim_time_s = float(objs_train['net'].t / b2.second)
        
        try:
            times_train_sec = np.array(times_train / b2.second, dtype=float)
        except Exception:
            times_train_sec = np.array(times_train, dtype=float)
        
        # Sumamos la hora actual
        times_adjusted_sec = times_train_sec + current_sim_time_s
        
        # Inyectamos los datos ajustados
        objs_train['input'].set_spikes(indices_train, times_adjusted_sec * b2.second)

        # E. Ejecutar simulación
        run_simulation(objs_train, duration=DURACION_TRAIN)

    # 2. GUARDADO
    print("\n💾 Guardando pesos y topología...")
    meta = {'description': 'ECG Training Sim-to-Real', 'epochs': NUM_EPOCHS}
    save_weights(objs_train, filename=WEIGHTS_FILE, metadata=meta)
    print("✅ Entrenamiento finalizado y guardado.")    
    # === AÑADE ESTAS LÍNEAS AQUÍ ===
    del objs_train  # Borramos las referencias a los objetos antiguos
    gc.collect()    # Forzamos al recolector de basura a limpiar Brian2
    # ===============================
    #%%
    # ==========================================
    # FASE 2: VALIDACIÓN CLÍNICA (Mundo Real)
    # ==========================================
    print(f"\n[{'='*20} FASE 2: TEST CLÍNICO (MIT-BIH) {'='*20}]")

    # NOTA: Ya no necesitamos cargar la topología antes. 
    # load_weights se encargará de reconectar.

    # 2. Construir red de test (Cerebro congelado)
    print("🏗️ Reconstruyendo red (STDP Desactivado)...")
    objs_test = build_network(
        n_input=TOTAL_NEURONAS_INPUT, 
        n_output=TOTAL_NEURONAS_OUTPUT,
        spike_indices=[0],      
        spike_times=[0]*b2.ms, 
        learning_enabled=False,
        
        # --- CAMBIOS CLAVE PARA EVITAR EL ERROR ---
        connectivity_prob=0,   # <--- Pone 0 cables al principio (Lienzo en blanco)
        topology_data=None     # <--- No cargues topología aquí, deja que load_weights lo haga
    )
    # 3. Cargar Valores Y Conexiones
    print("🧠 Restaurando memoria sináptica dual...")
    # Esto ahora conecta Sano y Arritmia por separado
    load_weights(objs_test, WEIGHTS_FILE) 
    
    # 4. Diagnóstico de pesos
    print("   🔎 Inspeccionando especialización...")
    if 'synapses_sano' in objs_test:
        if len(objs_test['synapses_sano']) > 0:
            w_s = objs_test['synapses_sano'].w[:]
            # CAMBIO: :.5f para ver 5 decimales
            print(f"      - Equipo Sano: Max={np.max(w_s):.5f} V") 
            
    if 'synapses_arr' in objs_test:
        if len(objs_test['synapses_arr']) > 0:
            w_a = objs_test['synapses_arr'].w[:]
            # CAMBIO: :.5f
            print(f"      - Equipo Arritmia: Max={np.max(w_a):.5f} V")
    
   # === SOLUCIÓN AL WARNING ===
    # Le decimos a Brian2: "Inicializa la red y prepárala".
    # Al correr 0 ms, Brian2 construye internamente la red, verifica conexiones
    # y marca todos los objetos como "USADOS".
    print("⚙️ Validando integridad de la red auxiliar...")
    objs_test['net'].run(0*b2.ms)
    
    # 5. Cargar Pacientes Reales Y VISUALIZAR
    loader = RealECGLoader(n_input=TOTAL_NEURONAS_INPUT)
    
    # --- CASO A: PACIENTE SANO ---
    print("\n🏥 Cargando Paciente Sano (MIT-BIH 115)...")
    # 1. Recuperamos la señal cruda para pintar
    raw_signal_100, _, fs_100 = loader.load_mit_bih_data('115', duration_sec=DURACION_TEST/b2.second)
    # 2. Generamos los spikes
    indices_healthy, times_healthy = loader.ecg_to_spikes(
        input_data='115', 
        duration_sec=DURACION_TEST/b2.second
    )
    loader.validate_detection(times_healthy/b2.second, '115')
    # 3. ¡PINTAMOS LA VALIDACIÓN! 🖌️
    from src.visualization import plot_ecg_validation
    plot_ecg_validation(raw_signal_100, fs_100, times_healthy, title="Validación Paciente 115 (Sano)")
    
    
    # --- CASO B: PACIENTE ARRÍTMICO ---
    print("💔 Cargando Paciente Arrítmico (MIT-BIH 203)...")
    raw_signal_200, _, fs_200 = loader.load_mit_bih_data('203', duration_sec=DURACION_TEST/b2.second)
    
    indices_arrhythmia, times_arrhythmia = loader.ecg_to_spikes(
        input_data='203', 
        duration_sec=DURACION_TEST/b2.second
    )
    loader.validate_detection(times_arrhythmia/b2.second, '203')
    
    plot_ecg_validation(raw_signal_200, fs_200, times_arrhythmia, title="Validación Paciente 203 (Arritmia)")
    
# 6. Ejecutar test
    print("\n🩺 Ejecutando diagnóstico...")
    
    # --- AJUSTE MAESTRO DE PARÁMETROS ---
    
    params = {
        'n_input': TOTAL_NEURONAS_INPUT,
        'n_output': TOTAL_NEURONAS_OUTPUT,
        'tau_sano': 5 * b2.ms,       
        'tau_arritmia': 100 * b2.ms,
        'w_inhib': 800 * b2.mV,       
        
        # --- NUEVOS UMBRALES DIVIDIDOS ---
        'v_thresh_sano': -20 * b2.mV,  # Sensible (Gatillo rápido)
        'v_thresh_arr':  -30 * b2.mV,  # Duro (Solo dispara si acumula mucho)
        
        # 'v_threshold': -53 * b2.mV  <--- ESTE YA NO SE USA, BORRALO O COMENTALO
    }
    
    # 2. EJECUCIÓN DEL DIAGNÓSTICO (Sin dopaje manual)
    print("\n🩺 Ejecutando diagnóstico clínico real...")
    
    stats, res_healthy, res_arrhythmia = compare_recognition(
        params,
        pattern_trained=(indices_healthy, times_healthy),     
        pattern_novel=(indices_arrhythmia, times_arrhythmia)
    )
    
   # 7. GRÁFICA FINAL
    # Para la gráfica visual, puedes pintar AMBOS umbrales o solo uno de referencia.
    # Pero para el informe numérico (lo importante), usamos los dos.
    
    u_sano = params.get('v_thresh_sano', -15*b2.mV) / b2.mV
    u_arr  = params.get('v_thresh_arr',  -35*b2.mV) / b2.mV
    
    # 2. Gráfica (Usamos el del sano como referencia visual principal)
    plot_recognition_comparison(res_healthy, res_arrhythmia, v_threshold_mv=u_sano)
    
    # 3. Informe (Pasamos LOS DOS umbrales para que cuente bien)
    print_diagnosis_report(res_healthy, res_arrhythmia, threshold_sano=u_sano, threshold_arr=u_arr)
    
    print("\n✅ PROYECTO COMPLETADO.")
