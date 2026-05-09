"""
Módulo de evaluación y visualización de resultados.

Este módulo genera todas las figuras y reportes que necesitas
para tu tesis. Cada función produce una figura lista para
exportar como imagen de alta resolución.

=== FIGURAS PARA LA TESIS ===

1. Confusion Matrix (por modelo):
   Muestra qué géneros se confunden entre sí. Esperarás ver
   confusiones entre rock/country y jazz/blues porque comparten
   características sonoras.

2. Curvas de entrenamiento (loss y accuracy vs epochs):
   Muestra si el modelo converge bien o tiene overfitting.
   Si train_loss baja pero val_loss sube = overfitting.
   Si ambas bajan juntas = buen entrenamiento.

3. Classification Report:
   Tabla con precision, recall y F1 por género.
   Precision = de los que predije como jazz, ¿cuántos son jazz?
   Recall = de todos los jazz reales, ¿cuántos detecté?
   F1 = media armónica de ambos.

4. Comparativa CNN vs CRNN:
   Gráfico de barras comparando métricas de ambos modelos.
   Esto es el corazón del análisis de tu tesis.

5. Distribución de métricas por fold:
   Box plot mostrando la variabilidad de las métricas across
   folds. Demuestra robustez del modelo.
"""
import logging

from config import ProjectConfig

logger = logging.getLogger(__name__)


# ===================================================================
# SECCIÓN 1: Confusion Matrix
# ===================================================================
# TODO: Implementar plot_confusion_matrix()
#   - Recibe: y_true, y_pred, class_names, title, save_path
#   - Usar sklearn.metrics.confusion_matrix
#   - Visualizar con seaborn.heatmap:
#       - annot=True para mostrar números
#       - fmt='d' para enteros
#       - cmap='Blues' (profesional para tesis)
#       - xticklabels y yticklabels con los nombres de géneros
#   - Agregar xlabel="Predicción", ylabel="Real"
#   - plt.tight_layout() antes de guardar
#   - Guardar como PNG 300dpi (calidad de tesis)
#
# TIP: Normalizar la confusion matrix dividiendo cada fila por
# su suma te da porcentajes, que son más fáciles de interpretar
# cuando las clases tienen diferente tamaño.


# ===================================================================
# SECCIÓN 2: Curvas de entrenamiento
# ===================================================================
# TODO: Implementar plot_training_curves()
#   - Recibe: history (dict del model.fit), fold_number, save_path
#   - Crear figura con 2 subplots (1 fila, 2 columnas):
#       - Subplot 1: Loss (train vs validation)
#       - Subplot 2: Accuracy (train vs validation)
#   - Usar colores consistentes:
#       - Train: azul sólido
#       - Validation: naranja con línea punteada
#   - Agregar leyenda, grid suave, títulos claros
#   - Si hay ReduceLROnPlateau, marcar los puntos donde bajó el LR
#   - Guardar como PNG 300dpi
#
# NOTA: Si el modelo tiene EarlyStopping, la curva se cortará
# antes de llegar a EPOCHS. Eso es normal y bueno.


# ===================================================================
# SECCIÓN 3: Classification Report
# ===================================================================
# TODO: Implementar generate_classification_report()
#   - Recibe: y_true, y_pred, class_names
#   - Usar sklearn.metrics.classification_report
#   - Versión texto: para imprimir en consola
#   - Versión dict: para generar tabla/gráfico
#   - Retornar ambas versiones
#
# TODO: Implementar plot_per_class_metrics()
#   - Gráfico de barras agrupadas: precision, recall, F1 por clase
#   - Esto muestra visualmente dónde el modelo es fuerte y débil
#   - Usar colores distintos para precision, recall, F1
#   - Agregar línea horizontal en el promedio macro


# ===================================================================
# SECCIÓN 4: Comparativa entre modelos
# ===================================================================
# TODO: Implementar plot_model_comparison()
#   - Recibe: dict con métricas de CNN y CRNN
#   - Gráfico de barras lado a lado
#   - Métricas: Accuracy, F1-macro, Precision-macro, Recall-macro
#   - Agregar barras de error (std de los folds)
#   - Esto es la figura más importante de la sección de resultados


# ===================================================================
# SECCIÓN 5: Análisis por fold
# ===================================================================
# TODO: Implementar plot_fold_distribution()
#   - Box plot con la distribución de cada métrica across folds
#   - Muestra variabilidad y robustez del modelo
#   - Si la varianza es alta, sugiere que el modelo es sensible
#     a la partición de datos (común con datasets pequeños)
#
# TODO: Implementar plot_fold_comparison()
#   - Gráfico de líneas: métrica vs fold_number
#   - Permite identificar si algún fold es "outlier"


# ===================================================================
# SECCIÓN 6: Exportar resultados
# ===================================================================
# TODO: Implementar save_results_to_csv()
#   - Guarda todas las métricas en CSV para fácil análisis
#   - Columnas: fold, accuracy, f1_macro, precision_macro,
#     recall_macro, loss, epochs_trained
#   - Fila final: media ± std
#
# TODO: Implementar generate_latex_table()
#   - Genera tabla en formato LaTeX lista para copiar a tu tesis
#   - Formato: modelo | accuracy | F1-macro | precision | recall
#   - Con notación media ± std
#   - Esto te ahorra horas formateando tablas manualmente


# ===================================================================
# SECCIÓN 7: Benchmarking de tiempos
# ===================================================================
# TODO: Implementar benchmark_preprocessing()
#   - Mide tiempo total de preprocesar TODO el dataset
#   - Usa time.perf_counter() (más preciso que time.time())
#   - Reporta: tiempo total, tiempo promedio por archivo
#
# TODO: Implementar benchmark_training()
#   - Mide tiempo por epoch (extraer de history o con callback)
#   - Reporta: tiempo promedio por epoch, tiempo total de training
#   - Comparar CNN vs CRNN en velocidad de entrenamiento
#
# TODO: Implementar benchmark_inference()
#   - Mide tiempo de predicción para UNA muestra
#   - Repetir N veces (ej: 100) y promediar para estabilidad
#   - Excluir la primera predicción (warmup del GPU/modelo)
#   - Reporta: latencia promedio ± std por predicción
#   - Esto es clave si argumentas viabilidad de despliegue
#
# TODO: Implementar plot_time_comparison()
#   - Gráfico de barras: CNN vs CRNN en las 3 métricas de tiempo
#   - Complementa la comparativa de precisión con costo real
#
# TODO: Implementar generate_benchmark_table()
#   - Tabla LaTeX con: modelo | params | epoch_time | inference_time
#   - Junto con la tabla de métricas, da la foto completa
