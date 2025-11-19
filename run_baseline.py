import os
import traci
import pandas as pd
import numpy as np

def run_baseline(sim_dir):
    print("--- CORRIENDO BASELINE (SIN IA) ---")
    # Configuración idéntica al entrenamiento
    net_file = os.path.join(sim_dir, "TestLightsSogamosoNet.net.xml")
    route_file = os.path.join(sim_dir, "osm.passenger.trips.xml") 
    
    sumo_cmd = ["sumo", "-n", net_file, "-r", route_file,
                "--no-step-log", "true", "--waiting-time-memory", "1000",
                "--time-to-teleport", "300"]
    
    traci.start(sumo_cmd)
    
    metrics = []
    step = 0
    while step < 3600:
        traci.simulationStep()
        
        # Recolectar métricas compatibles con sumo-rl
        vehs = traci.vehicle.getIDList()
        if vehs:
            speeds = [traci.vehicle.getSpeed(v) for v in vehs]
            waitings = [traci.vehicle.getWaitingTime(v) for v in vehs]
            stopped = [1 for v in vehs if traci.vehicle.getSpeed(v) < 0.1]
            
            metrics.append({
                "step": step,
                "system_mean_speed": np.mean(speeds),
                "system_total_waiting_time": sum(waitings),
                "system_total_stopped": sum(stopped)
            })
        else:
            metrics.append({"step": step, "system_mean_speed": 0, "system_total_waiting_time": 0, "system_total_stopped": 0})
            
        step += 1
    
    traci.close()
    df = pd.DataFrame(metrics)
    df.to_csv("datos_baseline.csv", index=False)
    print("Baseline guardado en 'datos_baseline.csv'")

if __name__ == "__main__":
    run_baseline("./sumoData") # Ajusta tu ruta