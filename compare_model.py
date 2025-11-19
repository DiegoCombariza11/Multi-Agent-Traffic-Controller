import pandas as pd
import matplotlib.pyplot as plt

df_base = pd.read_csv("datos_baseline.csv")
df_ai = pd.read_csv("datos_IA_evaluacion_conn1_ep1.csv") 

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].plot(df_base['step'], df_base['system_mean_speed'], label='Sin IA (Baseline)', color='gray', linestyle='--')
axs[0].plot(df_ai['step'], df_ai['system_mean_speed'], label='Con IA (DQN)', color='green')
axs[0].set_title('Comparativa de Velocidad Promedio')
axs[0].set_ylabel('m/s')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(df_base['step'], df_base['system_total_waiting_time'], label='Sin IA (Baseline)', color='gray', linestyle='--')
axs[1].plot(df_ai['step'], df_ai['system_total_waiting_time'], label='Con IA (DQN)', color='blue')
axs[1].set_title('Comparativa de Tiempo de Espera Acumulado')
axs[1].set_ylabel('Segundos')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig("IA_vs_Baseline.png")
plt.show()