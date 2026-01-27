# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 20:08:42 2026
@author: ggv16
"""
import brian2 as b2
import numpy as np

# --- MÓDULOS PROPIOS ---
from src.network import build_network
from src.simulation import run_simulation  # <--- USAMOS TU NOMBRE REAL
from src.weight_manager import save_weights, load_weights, load_topology
from src.test_recognition import compare_recognition
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
        connectivity_prob=1.0   
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
            
        # Generamos con el modo elegido
        indices_train, times_train = generate_ecg_data(
            n_input=TOTAL_NEURONAS_INPUT,
            duration_ms=DURACION_TRAIN/b2.ms,
            bpm=current_bpm,
            mode=modo_actual,    # <--- Variable dinámica
            noise_level=0.05
        )
        
       
        # C. Feedback
        print(f"   🔄 Época {epoch+1}/{NUM_EPOCHS} | Ritmo: {current_bpm:.1f} BPM | Spikes: {len(indices_train)}")
        
        # D. CORRECCIÓN TEMPORAL (CRÍTICO PARA QUE FUNCIONE) ⏱️
        # Obtenemos la hora actual del reloj de la simulación
        current_sim_time = objs_train['net'].t
        
        # Sumamos la hora actual a los spikes para que ocurran AHORA
        times_adjusted = times_train + current_sim_time
        
        # Inyectamos los datos ajustados
        objs_train['input'].set_spikes(indices_train, times_adjusted)

        # E. Ejecutar simulación (Usando tu función run_simulation)
        run_simulation(objs_train, duration=DURACION_TRAIN)

    # 2. GUARDADO
    print("\n💾 Guardando pesos y topología...")
    meta = {'description': 'ECG Training Sim-to-Real', 'epochs': NUM_EPOCHS}
    save_weights(objs_train, filename=WEIGHTS_FILE, metadata=meta)
    print("✅ Entrenamiento finalizado y guardado.")    

    # ==========================================
    # FASE 2: VALIDACIÓN CLÍNICA (Mundo Real)
    # ==========================================
    print(f"\n[{'='*20} FASE 2: TEST CLÍNICO (MIT-BIH) {'='*20}]")

    # 1. Cargar Topología
    try:
        print("📂 Cargando topología...")
        indices_i, indices_j = load_topology(WEIGHTS_FILE)
    except FileNotFoundError:
        print(f"❌ No se encuentra {WEIGHTS_FILE}. Ejecuta Fase 1.")
        exit()

    # 2. Construir red de test (Cerebro congelado)
    print("🏗️ Reconstruyendo red (STDP Desactivado)...")
    objs_test = build_network(
        n_input=TOTAL_NEURONAS_INPUT, 
        n_output=TOTAL_NEURONAS_OUTPUT,
        spike_indices=[0],     
        spike_times=[0]*b2.ms, 
        learning_enabled=False, 
        topology_data=(indices_i, indices_j) 
    )
    
    # 3. Cargar Valores de Pesos
    print("🧠 Restaurando memoria sináptica...")
    load_weights(objs_test, WEIGHTS_FILE)
    
    
    # 4. Diagnóstico de pesos
    if 'synapses' in objs_test:
        pesos = objs_test['synapses'].w[:]
        mean_w = float(np.mean(pesos / b2.mV))
        max_w = float(np.max(pesos / b2.mV))
        print(f"   🔎 Estado: Max={max_w:.3f} mV | Medio={mean_w:.3f} mV")

    # 5. Cargar Pacientes Reales Y VISUALIZAR
    loader = RealECGLoader(n_input=TOTAL_NEURONAS_INPUT)
    
    # --- CASO A: PACIENTE SANO ---
    print("\n🏥 Cargando Paciente Sano (MIT-BIH 100)...")
    # 1. Recuperamos la señal cruda para pintar
    raw_signal_100, _, fs_100 = loader.load_mit_bih_data('100', duration_sec=DURACION_TEST/b2.second)
    # 2. Generamos los spikes
    indices_healthy, times_healthy = loader.ecg_to_spikes(
        record_name='100', 
        duration_sec=DURACION_TEST/b2.second, 
        use_annotations=False
    )
    loader.validate_detection(times_healthy/b2.second, '100')
    # 3. ¡PINTAMOS LA VALIDACIÓN! 🖌️
    from src.visualization import plot_ecg_validation
    plot_ecg_validation(raw_signal_100, fs_100, times_healthy, title="Validación Paciente 100 (Sano)")
    
    
    # --- CASO B: PACIENTE ARRÍTMICO ---
    print("💔 Cargando Paciente Arrítmico (MIT-BIH 200)...")
    raw_signal_200, _, fs_200 = loader.load_mit_bih_data('200', duration_sec=DURACION_TEST/b2.second)
    
    indices_arrhythmia, times_arrhythmia = loader.ecg_to_spikes(
        record_name='200', 
        duration_sec=DURACION_TEST/b2.second, 
        use_annotations=False
    )
    loader.validate_detection(times_arrhythmia/b2.second, '200')
    
    plot_ecg_validation(raw_signal_200, fs_200, times_arrhythmia, title="Validación Paciente 200 (Arritmia)")
    
    # 6. Ejecutar test
    print("\n🩺 Ejecutando diagnóstico...")
    stats, res_healthy, res_arrhythmia = compare_recognition(
        objs_test,
        pattern_trained=(indices_healthy, times_healthy),    
        pattern_novel=(indices_arrhythmia, times_arrhythmia), 
        duration=DURACION_TEST,
        pattern_size=TAMANO_PATRON
    )
    
    # 7. GRÁFICA FINAL
    print("\n📈 Generando informe cardiológico visual...")
    
    # AHORA LE PASAMOS EL UMBRAL EXACTO QUE USASTE EN NETWORK.PY
    plot_recognition_comparison(
        res_healthy, 
        res_arrhythmia,
        v_threshold_mv=-50.0,  # <--- AQUÍ CAMBIAS LA LÍNEA ROJA
        title="Diagnóstico SNN: Competencia de Equipos (Verde vs Rojo)"
    )
    
    print("\n✅ PROYECTO COMPLETADO.")