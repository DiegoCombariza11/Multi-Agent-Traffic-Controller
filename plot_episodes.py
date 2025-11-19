import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

def natural_sort_key(s):
    """Ordena archivos numéricamente (ep1, ep2, ep10...) en lugar de texto (ep1, ep10, ep2)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def plot_learning_curve(results_dir="resultados_sumo_rl"):
    pattern = os.path.join(results_dir, "*conn*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print("No encontré archivos CSV. Revisa la ruta.")
        return

    files.sort(key=natural_sort_key)
    
    rewards = []
    waiting_times = []
    speeds = []
    
    print(f"Procesando {len(files)} episodios...")

    for f in files:
        df = pd.read_csv(f)
        rewards.append(df['system_total_waiting_time'].sum() * -1) 
        waiting_times.append(df['system_total_waiting_time'].mean())
        speeds.append(df['system_mean_speed'].mean())

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    axs[0].plot(range(1, len(files)+1), speeds, marker='o', color='green')
    axs[0].set_title('Velocidad Promedio por Episodio (Debe subir)')
    axs[0].set_ylabel('m/s')
    axs[0].grid(True)

    # Espera
    axs[1].plot(range(1, len(files)+1), waiting_times, marker='o', color='red')
    axs[1].set_title('Tiempo de Espera Promedio (Debe bajar)')
    axs[1].set_ylabel('Segundos')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("curva_aprendizaje.png")
    print("Gráfica guardada: curva_aprendizaje.png")
    plt.show()

if __name__ == "__main__":
    plot_learning_curve("./sumoData")