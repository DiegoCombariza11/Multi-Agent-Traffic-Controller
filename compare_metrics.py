"""Script para comparar métricas entre baseline, DQN y DQN+Agentes Regionales.

Genera comparaciones de:
1. Velocidad promedio del sistema
2. Tiempo de espera total
3. Total de vehículos parados
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_data(baseline_path: str, dqn_path: str, orchestrator_path: str):
    """Carga los 3 archivos CSV."""
    baseline = pd.read_csv(baseline_path)
    dqn = pd.read_csv(dqn_path)
    orchestrator = pd.read_csv(orchestrator_path)
    
    # Limpiar nombres de columnas
    for df in [baseline, dqn, orchestrator]:
        df.columns = df.columns.str.strip()
    
    return baseline, dqn, orchestrator


def plot_mean_speed(baseline: pd.DataFrame, dqn: pd.DataFrame, 
                    orchestrator: pd.DataFrame, output_dir: str):
    """Gráfico de velocidad promedio a lo largo del tiempo."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(baseline['step'], baseline['system_mean_speed'], 
           label='Baseline (Sin IA)', color='#d62728', linewidth=2, alpha=0.8)
    ax.plot(dqn['step'], dqn['system_mean_speed'], 
           label='DQN Solo', color='#ff7f0e', linewidth=2, alpha=0.8)
    ax.plot(orchestrator['step'], orchestrator['system_mean_speed'], 
           label='DQN + Agentes Regionales', color='#2ca02c', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel('Mean Speed (m/s)', fontsize=12)
    ax.set_title('System Mean Speed Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    output_path = os.path.join(output_dir, 'comparison_mean_speed.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_waiting_time(baseline: pd.DataFrame, dqn: pd.DataFrame, 
                     orchestrator: pd.DataFrame, output_dir: str):
    """Gráfico de tiempo de espera total a lo largo del tiempo."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(baseline['step'], baseline['system_total_waiting_time'], 
           label='Baseline (Sin IA)', color='#d62728', linewidth=2, alpha=0.8)
    ax.plot(dqn['step'], dqn['system_total_waiting_time'], 
           label='DQN Solo', color='#ff7f0e', linewidth=2, alpha=0.8)
    ax.plot(orchestrator['step'], orchestrator['system_total_waiting_time'], 
           label='DQN + Agentes Regionales', color='#2ca02c', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel('Total Waiting Time (s)', fontsize=12)
    ax.set_title('System Total Waiting Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    output_path = os.path.join(output_dir, 'comparison_waiting_time.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_stopped_vehicles(baseline: pd.DataFrame, dqn: pd.DataFrame, 
                         orchestrator: pd.DataFrame, output_dir: str):
    """Gráfico de vehículos parados a lo largo del tiempo."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(baseline['step'], baseline['system_total_stopped'], 
           label='Baseline (Sin IA)', color='#d62728', linewidth=2, alpha=0.8)
    ax.plot(dqn['step'], dqn['system_total_stopped'], 
           label='DQN Solo', color='#ff7f0e', linewidth=2, alpha=0.8)
    ax.plot(orchestrator['step'], orchestrator['system_total_stopped'], 
           label='DQN + Agentes Regionales', color='#2ca02c', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Simulation Step', fontsize=12)
    ax.set_ylabel('Total Stopped Vehicles', fontsize=12)
    ax.set_title('System Total Stopped Vehicles Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    output_path = os.path.join(output_dir, 'comparison_stopped_vehicles.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_summary_bars(baseline: pd.DataFrame, dqn: pd.DataFrame, 
                     orchestrator: pd.DataFrame, output_dir: str):
    """Gráfico de barras con promedios generales."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    
    # Calcular promedios
    metrics = {
        'Baseline': {
            'speed': baseline['system_mean_speed'].mean(),
            'waiting': baseline['system_total_waiting_time'].mean(),
            'stopped': baseline['system_total_stopped'].mean()
        },
        'DQN Solo': {
            'speed': dqn['system_mean_speed'].mean(),
            'waiting': dqn['system_total_waiting_time'].mean(),
            'stopped': dqn['system_total_stopped'].mean()
        },
        'DQN + Agentes': {
            'speed': orchestrator['system_mean_speed'].mean(),
            'waiting': orchestrator['system_total_waiting_time'].mean(),
            'stopped': orchestrator['system_total_stopped'].mean()
        }
    }
    
    models = list(metrics.keys())
    colors = ['#d62728', '#ff7f0e', '#2ca02c']
    
    # 1. Velocidad promedio
    speeds = [metrics[m]['speed'] for m in models]
    ax1.bar(models, speeds, color=colors, alpha=0.7)
    ax1.set_ylabel('Mean Speed (m/s)', fontsize=11)
    ax1.set_title('Average System Speed', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(speeds):
        ax1.text(i, v + 0.2, f'{v:.2f}', ha='center', fontweight='bold')
    
    # 2. Tiempo de espera
    waiting_times = [metrics[m]['waiting'] for m in models]
    ax2.bar(models, waiting_times, color=colors, alpha=0.7)
    ax2.set_ylabel('Total Waiting Time (s)', fontsize=11)
    ax2.set_title('Average Waiting Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(waiting_times):
        ax2.text(i, v + 20, f'{v:.0f}', ha='center', fontweight='bold')
    
    # 3. Vehículos parados
    stopped = [metrics[m]['stopped'] for m in models]
    ax3.bar(models, stopped, color=colors, alpha=0.7)
    ax3.set_ylabel('Total Stopped Vehicles', fontsize=11)
    ax3.set_title('Average Stopped Vehicles', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(stopped):
        ax3.text(i, v + 1, f'{v:.1f}', ha='center', fontweight='bold')
    
    # Rotar etiquetas x
    for ax in [ax1, ax2, ax3]:
        ax.set_xticklabels(models, rotation=15, ha='right')
    
    fig.tight_layout()
    output_path = os.path.join(output_dir, 'comparison_summary_bars.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def print_statistics(baseline: pd.DataFrame, dqn: pd.DataFrame, 
                    orchestrator: pd.DataFrame):
    """Imprime estadísticas comparativas."""
    print("\n" + "="*70)
    print("COMPARATIVE STATISTICS")
    print("="*70)
    
    datasets = {
        'Baseline (Sin IA)': baseline,
        'DQN Solo': dqn,
        'DQN + Agentes Regionales': orchestrator
    }
    
    for name, df in datasets.items():
        print(f"\n{name}:")
        print(f"  Mean Speed:          {df['system_mean_speed'].mean():8.2f} m/s  "
              f"(min: {df['system_mean_speed'].min():.2f}, max: {df['system_mean_speed'].max():.2f})")
        print(f"  Total Waiting Time:  {df['system_total_waiting_time'].mean():8.0f} s    "
              f"(min: {df['system_total_waiting_time'].min():.0f}, max: {df['system_total_waiting_time'].max():.0f})")
        print(f"  Stopped Vehicles:    {df['system_total_stopped'].mean():8.1f}      "
              f"(min: {df['system_total_stopped'].min():.0f}, max: {df['system_total_stopped'].max():.0f})")
    
    # Calcular mejoras
    print("\n" + "="*70)
    print("IMPROVEMENTS vs BASELINE")
    print("="*70)
    
    baseline_speed = baseline['system_mean_speed'].mean()
    baseline_waiting = baseline['system_total_waiting_time'].mean()
    baseline_stopped = baseline['system_total_stopped'].mean()
    
    for name, df in [('DQN Solo', dqn), ('DQN + Agentes Regionales', orchestrator)]:
        speed_diff = ((df['system_mean_speed'].mean() - baseline_speed) / baseline_speed) * 100
        waiting_diff = ((df['system_total_waiting_time'].mean() - baseline_waiting) / baseline_waiting) * 100
        stopped_diff = ((df['system_total_stopped'].mean() - baseline_stopped) / baseline_stopped) * 100
        
        print(f"\n{name}:")
        print(f"  Speed:          {speed_diff:+6.2f}% {'↑' if speed_diff > 0 else '↓'}")
        print(f"  Waiting Time:   {waiting_diff:+6.2f}% {'↓ (mejor)' if waiting_diff < 0 else '↑ (peor)'}")
        print(f"  Stopped:        {stopped_diff:+6.2f}% {'↓ (mejor)' if stopped_diff < 0 else '↑ (peor)'}")


def main():
    parser = argparse.ArgumentParser(description="Compare traffic metrics across models")
    parser.add_argument(
        '--baseline',
        type=str,
        default='./metrics/datos_baseline.csv',
        help='Path to baseline CSV'
    )
    parser.add_argument(
        '--dqn',
        type=str,
        default='./datos_IA_evaluacion_conn1_ep1.csv',
        help='Path to DQN evaluation CSV'
    )
    parser.add_argument(
        '--orchestrator',
        type=str,
        default='./metrics/orchestrator_eval_conn1_ep1.csv',
        help='Path to orchestrator evaluation CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./metrics',
        help='Directory to save plots'
    )
    args = parser.parse_args()
    
    print("Loading data...")
    baseline, dqn, orchestrator = load_data(args.baseline, args.dqn, args.orchestrator)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nGenerating plots...")
    plot_mean_speed(baseline, dqn, orchestrator, args.output_dir)
    plot_waiting_time(baseline, dqn, orchestrator, args.output_dir)
    plot_stopped_vehicles(baseline, dqn, orchestrator, args.output_dir)
    plot_summary_bars(baseline, dqn, orchestrator, args.output_dir)
    
    print_statistics(baseline, dqn, orchestrator)
    
    print("\n✓ All comparison plots generated successfully!")


if __name__ == "__main__":
    main()
