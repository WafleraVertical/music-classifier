"""
Pipeline de entrenamiento y validación cruzada.

=== ESTRATEGIA DE VALIDACIÓN ===

Se usa Stratified K-Fold Cross Validation (5 folds por defecto).

¿Por qué Stratified K-Fold y no un simple train/test split?

1. GTZAN tiene solo 100 muestras por género. Un split 80/20
   dejaría solo 20 muestras de test por clase. Cualquier
   resultado con tan pocas muestras tiene alta varianza.

2. K-Fold usa TODO el dataset como test (en diferentes folds),
   dando un estimado más robusto del rendimiento real.

3. "Stratified" garantiza que cada fold tenga la misma proporción
   de géneros que el dataset completo, evitando folds sesgados.

4. Al final reportas: media ± desviación estándar de las métricas
   across folds. Esto es mucho más convincente para una tesis
   que un solo número.

=== CALLBACKS ===

EarlyStopping:
    Monitorea val_loss. Si no mejora en 'patience' épocas, detiene
    el entrenamiento y restaura los mejores pesos. Esto previene
    overfitting: el modelo deja de entrenar justo cuando empieza
    a memorizar el training set.

ReduceLROnPlateau:
    Si val_loss no mejora en 'lr_reduce_patience' épocas, reduce
    el learning rate multiplicándolo por 'lr_reduce_factor'.
    Esto permite que el modelo "refine" su convergencia con pasos
    más pequeños cuando se acerca al óptimo.

ModelCheckpoint:
    Guarda el mejor modelo (por val_loss) de cada fold.
    Así no pierdes el mejor modelo aunque el training siga.

=== FLUJO POR FOLD ===

Para cada fold k de K:
    1. Separar datos en train y validation según los índices del fold.
    2. Crear generadores de datos (o arrays).
    3. Construir modelo desde cero (pesos aleatorios).
    4. Entrenar con callbacks.
    5. Evaluar en validation set del fold.
    6. Guardar métricas y mejor modelo.
    7. Calcular confusion matrix del fold.

Al final de todos los folds:
    - Promediar métricas: accuracy, F1-macro, precision, recall.
    - Reportar media ± std.
    - Generar confusion matrix agregada.
"""
import logging

from config import ProjectConfig

logger = logging.getLogger(__name__)


# ===================================================================
# SECCIÓN 1: Preparación de datos por fold
# ===================================================================
# TODO: Implementar create_stratified_kfold()
#   - Usar sklearn.model_selection.StratifiedKFold
#   - n_splits=config.training.k_folds
#   - shuffle=True, random_state=config.training.random_seed
#   - Retornar el objeto StratifiedKFold
#
# TODO: Implementar prepare_fold_data()
#   - Recibe X, y, train_indices, val_indices
#   - Separa datos según los índices
#   - Convierte etiquetas a one-hot encoding
#     (tf.keras.utils.to_categorical)
#   - Retorna X_train, X_val, y_train, y_val


# ===================================================================
# SECCIÓN 2: Callbacks
# ===================================================================
# TODO: Implementar create_callbacks()
#   - Recibe TrainingConfig y fold_number
#   - Crear EarlyStopping:
#       monitor='val_loss', patience=config.patience,
#       restore_best_weights=True, min_delta=config.min_delta
#   - Crear ReduceLROnPlateau:
#       monitor='val_loss', factor=config.lr_reduce_factor,
#       patience=config.lr_reduce_patience, min_lr=config.lr_min
#   - Crear ModelCheckpoint:
#       filepath=f"best_model_fold_{fold_number}.keras"
#       save_best_only=True, monitor='val_loss'
#   - Retornar lista de callbacks
#
# NOTA: Si quieres métricas custom como F1 por epoch, necesitas
# un callback custom. Ver SECCIÓN 5.


# ===================================================================
# SECCIÓN 3: Entrenamiento de un fold
# ===================================================================
# TODO: Implementar train_single_fold()
#   - Recibe: X_train, X_val, y_train, y_val, config, fold_number
#   - Construir modelo nuevo (build_cnn_model o build_crnn_model)
#   - Compilar modelo (compile_model)
#   - Crear callbacks
#   - Ejecutar model.fit()
#   - Retornar: model, history
#
# IMPORTANTE: El modelo se construye DESDE CERO en cada fold.
# No debes reutilizar pesos entre folds, porque eso sesgaría
# los resultados (el modelo ya "vio" datos del validation set
# de otro fold durante su training).


# ===================================================================
# SECCIÓN 4: Loop de validación cruzada
# ===================================================================
# TODO: Implementar run_cross_validation()
#   - Recibe: X, y, config, model_type ('cnn' o 'crnn')
#   - Para cada fold:
#       - Separar datos
#       - Entrenar
#       - Evaluar en val set
#       - Calcular métricas: accuracy, F1-macro, precision, recall
#       - Guardar history y métricas
#   - Al final:
#       - Calcular media y std de cada métrica across folds
#       - Imprimir resumen
#       - Retornar: dict con métricas por fold y promedio


# ===================================================================
# SECCIÓN 5: Callback custom para F1-Score (opcional)
# ===================================================================
# TODO: Implementar F1ScoreCallback (hereda de tf.keras.callbacks.Callback)
#   - En on_epoch_end:
#       - Hacer predicciones sobre validation set
#       - Calcular F1-macro con sklearn.metrics.f1_score
#       - Guardar en logs para que aparezca en history
#   - Esto es útil porque Keras por defecto no incluye F1-macro
#     como métrica built-in de manera confiable para multiclase.
#
# ALTERNATIVA: Usar tensorflow_addons.metrics.F1Score si prefieres
# no implementar el callback custom. Pero tfa puede tener
# problemas de compatibilidad con versiones recientes de TF.


# ===================================================================
# SECCIÓN 6: Entrenamiento final
# ===================================================================
# TODO: Implementar train_final_model()
#   - Después de validación cruzada, entrena un modelo final
#     con TODOS los datos de entrenamiento
#   - Usa los hiperparámetros que mejor funcionaron
#   - Este es el modelo que guardas para producción/demo
#   - Retorna modelo entrenado
