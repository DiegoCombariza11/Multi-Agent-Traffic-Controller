import os
import numpy as np
import config
import argparse
import sys

# Intentamos importar libsumo, si falla usamos traci standard
try:
    import libsumo as traci
except ImportError:
    import traci

def run_baseline(sim_dir, sumo_cfg):
    print("\n--- EJECUTANDO BASELINE (SIN I.A.) ---")
    print("Objetivo: Medir el rendimiento predeterminado de la red.")

    # Gestión de directorios (Igual que el worker)
    original_dir = os.getcwd()
    if not os.path.exists(sim_dir):
        print(f"Error: No existe el directorio {sim_dir}")
        return

    try:
        os.chdir(sim_dir)
        
        # Configuración para correr SIN GUI (rápido)
        # Creamos un objeto namespace dummy para reutilizar get_sumo_cmd
        dummy_args = argparse.Namespace(gui=False, sumo_cfg=sumo_cfg)
        cmd = config.get_sumo_cmd(dummy_args)
        
        traci.start(cmd)
        
        waiting_times = []
        speeds = []
        step = 0
        
        # Correr simulación hasta el límite definido en config
        MAX_STEPS = 3600 
        
        while traci.simulation.getMinExpectedNumber() > 0 and step < MAX_STEPS:
            traci.simulationStep()
            step += 1
            
            # Recolectar métricas del sistema completo
            # (Esto puede ser un poco lento, si tarda mucho reduce la frecuencia)
            vehs = traci.vehicle.getIDList()
            if vehs:
                # Velocidad promedio actual
                speeds.append(np.mean([traci.vehicle.getSpeed(v) for v in vehs]))
                # Espera acumulada de los vehículos presentes
                waiting_times.append(sum([traci.vehicle.getWaitingTime(v) for v in vehs]))

        arrived = traci.simulation.getArrivedNumber()
        traci.close()
        
        # Cálculo de promedios
        avg_speed = np.mean(speeds) if speeds else 0
        avg_wait_accum = np.mean(waiting_times) if waiting_times else 0
        
        print(f"\nRESULTADOS BASELINE:")
        print(f"-----------------------------")
        print(f"Vehículos Llegados: {arrived}")
        print(f"Velocidad Promedio: {avg_speed:.2f} m/s")
        print(f"Espera Promedio (Sistema): {avg_wait_accum:.2f}")
        print(f"-----------------------------")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_dir", type=str, default=config.DEFAULT_CONFIG["sim_dir"])
    parser.add_argument("--sumo_cfg", type=str, default=config.DEFAULT_CONFIG["sumo_cfg"])
    args = parser.parse_args()
    
    run_baseline(args.sim_dir, args.sumo_cfg)