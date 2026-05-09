"""
Punto de entrada principal del proyecto.

Ejecuta el pipeline completo:
    1. Configura el proyecto y crea directorios.
    2. Carga y preprocesa el dataset.
    3. Entrena el modelo CNN con validación cruzada.
    4. Entrena el modelo CRNN con validación cruzada.
    5. Compara ambos modelos.
    6. Genera figuras y reportes para la tesis.

Uso:
    python main.py

    # O para entrenar solo un modelo:
    python main.py --model cnn
    python main.py --model crnn

    # Para cambiar hiperparámetros desde CLI:
    python main.py --batch-size 64 --epochs 150 --lr 0.0001
"""
import argparse
import logging
import sys

from config import ProjectConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # TODO: Agregar FileHandler para guardar logs
    ],
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Clasificación de géneros musicales con CNN/CRNN"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["cnn", "crnn", "both"],
        help="Qué modelo entrenar (default: both)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Tamaño de batch (override config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Máximo de épocas (override config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate inicial (override config)",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Desactivar data augmentation",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directorio base de datos (override config)",
    )
    return parser.parse_args()


def apply_cli_overrides(
    config: ProjectConfig,
    args: argparse.Namespace,
) -> ProjectConfig:
    """Aplica overrides de CLI sobre la configuración."""
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.no_augmentation:
        config.augmentation.enable = False
    if args.data_dir is not None:
        from pathlib import Path
        config.paths.base_dir = Path(args.data_dir)
    return config


def main() -> None:
    """Pipeline principal."""
    args = parse_arguments()
    config = ProjectConfig()
    config = apply_cli_overrides(config, args)

    logger.info("=" * 60)
    logger.info("Clasificación de Géneros Musicales con CNN/CRNN")
    logger.info("=" * 60)

    # Paso 1: Crear directorios
    logger.info("Paso 1: Creando estructura de directorios...")
    config.paths.create_all()

    # Paso 2: Cargar y preprocesar datos
    logger.info("Paso 2: Cargando y preprocesando dataset...")
    # TODO: Llamar a load_dataset() de data_pipeline.py
    # X, y, label_map = load_dataset(config)
    # config.setup_num_classes(len(label_map))
    # config.setup_input_shapes()

    # Paso 3: Entrenar CNN
    if args.model in ("cnn", "both"):
        logger.info("Paso 3: Entrenando modelo CNN...")
        # TODO: Llamar a run_cross_validation() con model_type='cnn'
        # cnn_results = run_cross_validation(X, y, config, 'cnn')

    # Paso 4: Entrenar CRNN
    if args.model in ("crnn", "both"):
        logger.info("Paso 4: Entrenando modelo CRNN...")
        # TODO: Llamar a run_cross_validation() con model_type='crnn'
        # crnn_results = run_cross_validation(X, y, config, 'crnn')

    # Paso 5: Comparar modelos
    if args.model == "both":
        logger.info("Paso 5: Comparando modelos...")
        # TODO: Llamar a plot_model_comparison()
        # TODO: Llamar a generate_latex_table()

    # Paso 6: Generar reportes
    logger.info("Paso 6: Generando reportes finales...")
    # TODO: Llamar a save_results_to_csv()
    # TODO: Generar todas las figuras

    logger.info("=" * 60)
    logger.info("Pipeline completado exitosamente.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
