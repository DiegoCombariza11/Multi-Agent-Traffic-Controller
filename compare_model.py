import pandas as pd
import matplotlib.pyplot as plt

file_1 = "./graphics/datos_baseline.csv"          
file_2 = "./graphics/datos_IA_evaluacion_conn1_ep1_v3.csv"   
file_3 = "./graphics/datos_IA_evaluacion_conn1_ep1_v6.csv"  


label_1 = "Baseline (No AI)"
label_2 = "AI Model  V1"
label_3 = "AI Model  V2"



df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)
df3 = pd.read_csv(file_3)


plt.figure(figsize=(10, 6)) 


plt.plot(df1['step'], df1['system_mean_speed'], label=label_1, color='gray', linestyle='--', alpha=0.7)
plt.plot(df2['step'], df2['system_mean_speed'], label=label_2, color='green')
plt.plot(df3['step'], df3['system_mean_speed'], label=label_3, color='blue')


plt.title('System Mean Speed Comparison')
plt.xlabel('Simulation Step')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.savefig("Comparison_Mean_Speed.png") 
plt.show() 



plt.figure(figsize=(10, 6)) 


plt.plot(df1['step'], df1['system_total_waiting_time'], label=label_1, color='gray', linestyle='--', alpha=0.7)
plt.plot(df2['step'], df2['system_total_waiting_time'], label=label_2, color='green')
plt.plot(df3['step'], df3['system_total_waiting_time'], label=label_3, color='blue')


plt.title('System Total Waiting Time Comparison')
plt.xlabel('Simulation Step')
plt.ylabel('Total Waiting Time (seconds)')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.savefig("Comparison_Waiting_Time.png") 
plt.show()