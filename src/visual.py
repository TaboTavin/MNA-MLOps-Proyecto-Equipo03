"""
Módulo de visualización para comparar modelos de MLflow.
Genera gráficas de barras, radar, evolución temporal y tablas comparativas.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os


class ModelVisualizer:
    """Genera visualizaciones para comparar métricas de modelos registrados en MLflow."""
    
    def __init__(self, output_dir: str = "reports"):
        """Inicializa el visualizador con el directorio de salida especificado."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_model_comparison(self, comparison_data: List[Dict], metric_name: str = 'accuracy', 
                             top_n: int = 10, save: bool = True, show: bool = False):
        """Genera gráfica de barras horizontales comparando modelos por una métrica específica."""
        if not comparison_data:
            return None
        
        sorted_data = sorted(comparison_data, key=lambda x: x['metrics'][metric_name], reverse=True)[:top_n]
        
        model_names = [run['tags'].get('mlflow.runName', 'Unknown') for run in sorted_data]
        metric_values = [run['metrics'][metric_name] for run in sorted_data]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(model_names, metric_values, color='steelblue')
        
        # Resaltar el mejor modelo
        bars[0].set_color('darkgreen')
        
        ax.set_xlabel(metric_name.capitalize())
        ax.set_title(f'Comparación de Modelos - {metric_name.capitalize()}')
        ax.invert_yaxis()
        
        # Agregar valores en las barras
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            ax.text(value, i, f' {value:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        filepath = None
        if save:
            filepath = os.path.join(self.output_dir, f'model_comparison_{metric_name}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath
    
    def plot_metrics_distribution(self, comparison_data: List[Dict], metrics: List[str], 
                                  save: bool = True, show: bool = False):
        """Genera gráfica de barras agrupadas con múltiples métricas para cada modelo."""
        if not comparison_data:
            return None
        
        # Filtrar solo runs que tengan todas las métricas
        valid_data = [run for run in comparison_data 
                     if all(metric in run['metrics'] for metric in metrics)]
        
        if not valid_data:
            return None
        
        model_names = [run['tags'].get('mlflow.runName', 'Unknown')[:15] for run in valid_data]
        
        x = np.arange(len(model_names))
        width = 0.8 / len(metrics)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, metric in enumerate(metrics):
            values = [run['metrics'][metric] for run in valid_data]
            offset = width * i - (width * len(metrics) / 2)
            ax.bar(x + offset, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Modelos')
        ax.set_ylabel('Valor')
        ax.set_title('Comparación de Múltiples Métricas')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        filepath = None
        if save:
            filepath = os.path.join(self.output_dir, 'metrics_distribution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath
    
    def plot_top_models_radar(self, comparison_data: List[Dict], metrics: List[str], 
                             top_n: int = 5, save: bool = True, show: bool = False):
        """Genera gráfica de radar comparando los mejores modelos en múltiples métricas."""
        if not comparison_data or not metrics:
            return None
        
        # Filtrar runs con todas las métricas
        valid_data = [run for run in comparison_data 
                     if all(metric in run['metrics'] for metric in metrics)]
        
        if not valid_data or len(valid_data) < top_n:
            return None
        
        # Ordenar por primera métrica y tomar top N
        sorted_data = sorted(valid_data, key=lambda x: x['metrics'][metrics[0]], reverse=True)[:top_n]
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, top_n))
        
        for idx, run in enumerate(sorted_data):
            model_name = run['tags'].get('mlflow.runName', 'Unknown')
            values = [run['metrics'][metric] for metric in metrics]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title(f'Top {top_n} Modelos - Comparación Multimétrica', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        
        filepath = None
        if save:
            filepath = os.path.join(self.output_dir, 'top_models_radar.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath
    
    def plot_experiment_evolution(self, comparison_data: List[Dict], metric_name: str = 'accuracy',
                                 save: bool = True, show: bool = False):
        """Genera gráfica de línea mostrando la evolución temporal de una métrica."""
        if not comparison_data:
            return None
        
        # Ordenar por timestamp
        sorted_data = sorted(comparison_data, key=lambda x: x['start_time'])
        
        model_names = [run['tags'].get('mlflow.runName', 'Unknown')[:15] for run in sorted_data]
        metric_values = [run['metrics'][metric_name] for run in sorted_data]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(range(len(metric_values)), metric_values, marker='o', linewidth=2, markersize=8)
        
        # Resaltar el mejor valor
        best_idx = np.argmax(metric_values)
        ax.plot(best_idx, metric_values[best_idx], marker='*', markersize=20, 
                color='red', label='Mejor modelo')
        
        ax.set_xlabel('Orden de ejecución')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f'Evolución del Experimento - {metric_name.capitalize()}')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        filepath = None
        if save:
            filepath = os.path.join(self.output_dir, f'experiment_evolution_{metric_name}.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath
    
    def plot_metrics_comparison_table(self, metrics_data: List[Dict], 
                                      save: bool = True, show: bool = False):
        """Genera gráfica de barras agrupadas comparando múltiples métricas entre modelos."""
        if not metrics_data:
            return None
        
        # Extraer nombres de modelos y métricas
        models = [d['model'] for d in metrics_data]
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Verificar que todas las métricas existen
        available_metrics = [m for m in metrics if all(m in d for d in metrics_data)]
        
        if not available_metrics:
            return None
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, metric in enumerate(available_metrics):
            values = [d[metric] for d in metrics_data]
            offset = width * (i - len(available_metrics)/2 + 0.5)
            bars = ax.bar(x + offset, values, width, label=metric.capitalize(), color=colors[i])
            
            # Agregar valores encima de las barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Modelos', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        filepath = None
        if save:
            filepath = os.path.join(self.output_dir, 'metrics_comparison_table.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return filepath
