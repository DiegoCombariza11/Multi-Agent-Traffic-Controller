import argparse
import os

# Configuración por defecto
DEFAULT_CONFIG = {
    "episodes": 20,
    "batch_size": 6,
    "alpha": 0.05,
    "gamma": 0.95,
    "epsilon_start": 1.0,
    "epsilon_decay": 0.98,
    "min_epsilon": 0.05,
    "sumo_cfg": "SumoConfigSim.sumocfg", # Nombre del archivo, no la ruta completa
    "gui": False,
    "sim_dir": "./sumoData",       # Directorio donde están los archivos .net.xml, .rou.xml
    "output_dir": "./outputData" # Directorio donde se guardarán los CSV
}

def get_arguments():
    parser = argparse.ArgumentParser(description="Entrenamiento RL Semáforos SUMO")

    # Argumentos de Directorios
    parser.add_argument("--sim_dir", type=str, default=DEFAULT_CONFIG["sim_dir"], 
                        help="Ruta al directorio que contiene la simulación de SUMO")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"], 
                        help="Ruta donde se guardarán los archivos CSV de resultados")
    parser.add_argument("--sumo_cfg", type=str, default=DEFAULT_CONFIG["sumo_cfg"],
                        help="Nombre del archivo .sumocfg (debe estar dentro de sim_dir)")

    # Hiperparámetros
    parser.add_argument("--episodes", type=int, default=DEFAULT_CONFIG["episodes"], help="Total de episodios")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"], help="Cores/Workers")
    parser.add_argument("--alpha", type=float, default=DEFAULT_CONFIG["alpha"], help="Tasa de aprendizaje")
    parser.add_argument("--gamma", type=float, default=DEFAULT_CONFIG["gamma"], help="Factor de descuento")
    parser.add_argument("--epsilon_decay", type=float, default=DEFAULT_CONFIG["epsilon_decay"], help="Decaimiento Epsilon")
    parser.add_argument("--gui", action="store_true", help="Activar GUI")

    parser.add_argument("--epsilon_start", type=float, default=DEFAULT_CONFIG["epsilon_start"], help="Epsilon inicial")
    parser.add_argument("--min_epsilon", type=float, default=DEFAULT_CONFIG["min_epsilon"], help="Epsilon mínimo")
    
    args = parser.parse_args()
    return args

def get_sumo_cmd(args):
    """Genera el comando de SUMO. Nota: La ruta se maneja cambiando el directorio de trabajo en el worker."""
    binary = "sumo-gui" if args.gui else "sumo"
    cmd = [
        binary,
        "-c", args.sumo_cfg, 
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--duration-log.disable", "true",
        "--waiting-time-memory", "1000",
        "--time-to-teleport", "300"
    ]
    return cmd