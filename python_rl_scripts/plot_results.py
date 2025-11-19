import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import config

def plot_training(output_dir):
    csv_path = os.path.join(output_dir, "metrics_global.csv")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: No se encuentra el archivo {csv_path}")
        print("Asegúrate de haber ejecutado el entrenamiento primero.")
        return

    print(f"Cargando datos de: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Calcular media móvil para suavizar las líneas (tendencia)
    window = max(1, int(len(df)*0.1)) # 10% de los datos
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Resultados del Entrenamiento ({len(df)} Episodios)', fontsize=16)

    # 1. Recompensa Total
    axs[0, 0].plot(df['Episode'], df['Total_Reward'], alpha=0.3, color='gray', label='Crudo')
    axs[0, 0].plot(df['Episode'], df['Total_Reward'].rolling(window).mean(), color='blue', linewidth=2, label='Tendencia')
    axs[0, 0].set_title('Utilidad Global (Recompensa)')
    axs[0, 0].set_ylabel('Recompensa (Mayor es mejor)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Vehículos Llegados (Throughput)
    axs[0, 1].plot(df['Episode'], df['Arrived_Vehicles'], alpha=0.3, color='orange')
    axs[0, 1].plot(df['Episode'], df['Arrived_Vehicles'].rolling(window).mean(), color='red', linewidth=2)
    axs[0, 1].set_title('Vehículos Completados (Throughput)')
    axs[0, 1].set_ylabel('Cantidad de Vehículos')
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Velocidad Promedio
    axs[1, 0].plot(df['Episode'], df['Avg_Speed'], color='green', alpha=0.6)
    axs[1, 0].plot(df['Episode'], df['Avg_Speed'].rolling(window).mean(), color='darkgreen', linewidth=2)
    axs[1, 0].set_title('Velocidad Promedio de la Red')
    axs[1, 0].set_ylabel('m/s')
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Epsilon y RMS
    ax4 = axs[1, 1]
    ax4.plot(df['Episode'], df['Epsilon'], color='purple', linestyle='--', label='Exploración (Epsilon)')
    ax4.set_ylabel('Epsilon', color='purple')
    
    # Eje secundario para RMS
    ax4b = ax4.twinx()
    ax4b.plot(df['Episode'], df['RMS'], color='brown', alpha=0.5, label='Error RMS')
    ax4b.set_ylabel('RMS Error', color='brown')
    ax4.set_title('Exploración y Error de Aprendizaje')
    
    plt.tight_layout()
    
    # Guardar y Mostrar
    save_path = os.path.join(output_dir, "graficas_entrenamiento.png")
    plt.savefig(save_path)
    print(f"Gráficas guardadas en: {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=config.DEFAULT_CONFIG["output_dir"])
    args = parser.parse_args()
    plot_training(args.output_dir)