"""
Configuración centralizada de hiperparámetros y rutas del proyecto.

Este módulo contiene TODOS los parámetros configurables del proyecto.
La idea es que si cambias de dataset (por ejemplo, de GTZAN a FMA o
a uno propio), solo necesitas modificar ESTE archivo.

=== GUÍA DE HIPERPARÁMETROS ===

SAMPLE_RATE (22050 Hz):
    Frecuencia de muestreo estándar en análisis musical.
    22050 captura frecuencias hasta ~11 kHz (teorema de Nyquist),
    suficiente para la mayoría de características musicales.
    Si tu dataset tiene audio de mayor calidad, podrías subir a 44100.

DURATION (30 segundos):
    GTZAN tiene clips de 30s. Si cambias de dataset, ajusta esto.
    Clips más cortos = más muestras pero menos contexto temporal.
    Clips más largos = mejor contexto pero más memoria GPU.

N_MELS (128):
    Número de bandas de frecuencia Mel. Más bandas = más resolución
    en frecuencia pero espectrogramas más grandes.
    64: resolución baja, rápido de entrenar.
    128: balance estándar (recomendado).
    256: alta resolución, útil si tu audio tiene matices sutiles.

HOP_LENGTH (512):
    Salto entre ventanas FFT. Controla la resolución temporal.
    Menor hop = más frames temporales = espectrogramas más anchos.
    512 con sr=22050 da ~43 frames/segundo, buen balance.

N_FFT (2048):
    Tamaño de la ventana FFT. Controla la resolución en frecuencia
    de cada frame individual.
    2048 es el estándar. Subir a 4096 mejora resolución frecuencial
    pero pierde resolución temporal.

CONV_BLOCKS:
    Lista de diccionarios, cada uno define un bloque convolucional.
    Cada bloque tiene: filters, kernel_size, pool_size.
    El patrón clásico es duplicar filtros conforme avanzas en
    profundidad: 32 → 64 → 128 → 256.
    Más bloques = más capacidad pero más riesgo de sobreajuste.

DROPOUT_RATE (0.3 a 0.5):
    Probabilidad de "apagar" neuronas durante entrenamiento.
    0.3: regularización suave, bueno para datasets grandes.
    0.5: regularización fuerte, mejor para datasets pequeños (GTZAN).
    Si ves overfitting, sube este valor.

LEARNING_RATE (1e-3):
    Tasa de aprendizaje inicial para Adam.
    1e-3: estándar para Adam, buen punto de partida.
    1e-4: más conservador, útil si el training es inestable.
    ReduceLROnPlateau lo bajará automáticamente si el val_loss
    deja de mejorar.

BATCH_SIZE (32):
    Muestras por paso de gradiente.
    16: más ruido en gradientes, puede generalizar mejor.
    32: balance estándar.
    64: gradientes más estables pero necesita más memoria.

EPOCHS (100):
    Máximo de épocas. Early stopping detendrá el entrenamiento
    antes si val_loss no mejora en PATIENCE épocas.

PATIENCE (10):
    Épocas sin mejora antes de detener entrenamiento.
    Valor bajo (5): agresivo, puede parar antes de converger.
    Valor alto (15-20): más exploración, riesgo de sobreajuste.

K_FOLDS (5):
    Número de folds para validación cruzada estratificada.
    5 es estándar. 10 es más robusto pero 2x más lento.

=== MÉTRICAS ===

Se usa F1-Score Macro como métrica principal porque:
1. GTZAN tiene clases balanceadas (100 por género), pero si cambias
   de dataset podrían estar desbalanceadas. F1-macro trata TODAS
   las clases con igual importancia sin importar su tamaño.
2. Accuracy puede ser engañosa con clases desbalanceadas.
3. F1-macro combina precision y recall, dando una vista más
   completa del rendimiento por clase.

Accuracy se reporta como métrica secundaria por ser intuitiva.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AudioConfig:
    """Parámetros de procesamiento de audio."""

    sample_rate: int = 22_050
    duration: float = 30.0
    n_mels: int = 128
    hop_length: int = 512
    n_fft: int = 2048
    n_mfcc: int = 13  # Solo si decides usar MFCCs como feature adicional

    @property
    def target_samples(self) -> int:
        """Número total de muestras por clip."""
        return int(self.sample_rate * self.duration)

    @property
    def spectrogram_width(self) -> int:
        """Ancho aproximado del espectrograma (frames temporales)."""
        return int(self.target_samples / self.hop_length) + 1


@dataclass
class AugmentationConfig:
    """Parámetros de data augmentation.

    Cada probabilidad controla qué tan frecuente se aplica esa
    transformación durante el entrenamiento. 0.0 = nunca, 1.0 = siempre.
    """

    enable: bool = True
    noise_factor: float = 0.005
    noise_probability: float = 0.5
    time_stretch_range: tuple[float, float] = (0.8, 1.2)
    time_stretch_probability: float = 0.3
    pitch_shift_range: tuple[int, int] = (-2, 2)
    pitch_shift_probability: float = 0.3
    time_shift_max_ms: int = 500
    time_shift_probability: float = 0.5


@dataclass
class ConvBlockConfig:
    """Configuración de un bloque convolucional individual."""

    filters: int = 32
    kernel_size: tuple[int, int] = (3, 3)
    pool_size: tuple[int, int] = (2, 2)
    use_batch_norm: bool = True


@dataclass
class ModelConfig:
    """Configuración del modelo CNN.

    CONV_BLOCKS define la arquitectura convolucional. El patrón
    estándar duplica filtros por bloque: 32 → 64 → 128 → 256.
    Puedes agregar o quitar bloques modificando esta lista.
    """

    conv_blocks: list[ConvBlockConfig] = field(default_factory=lambda: [
        ConvBlockConfig(filters=32, kernel_size=(3, 3), pool_size=(2, 2)),
        ConvBlockConfig(filters=64, kernel_size=(3, 3), pool_size=(2, 2)),
        ConvBlockConfig(filters=128, kernel_size=(3, 3), pool_size=(2, 2)),
        ConvBlockConfig(filters=256, kernel_size=(3, 3), pool_size=(2, 2)),
    ])
    dense_units: list[int] = field(default_factory=lambda: [256, 128])
    dropout_rate: float = 0.5
    use_global_avg_pooling: bool = True
    activation: str = "relu"
    output_activation: str = "softmax"
    # Se calculará dinámicamente según el dataset
    num_classes: Optional[int] = None
    # Dimensiones de entrada (se calculan desde AudioConfig)
    input_shape: Optional[tuple[int, int, int]] = None


@dataclass
class CRNNConfig:
    """Configuración del modelo CRNN (contrapropuesta).

    La idea del CRNN es usar bloques CNN para extraer features
    espaciales del espectrograma, y luego una capa recurrente
    (GRU o LSTM) para capturar dependencias temporales que una
    CNN pura pierde.

    Esto es especialmente útil para música porque el ritmo,
    los cambios de sección y las progresiones son fenómenos
    inherentemente secuenciales.
    """

    conv_blocks: list[ConvBlockConfig] = field(default_factory=lambda: [
        ConvBlockConfig(filters=32, kernel_size=(3, 3), pool_size=(2, 2)),
        ConvBlockConfig(filters=64, kernel_size=(3, 3), pool_size=(2, 2)),
        ConvBlockConfig(filters=128, kernel_size=(3, 3), pool_size=(2, 2)),
    ])
    rnn_type: str = "gru"  # "gru" o "lstm"
    rnn_units: int = 128
    rnn_bidirectional: bool = True
    rnn_dropout: float = 0.3
    dense_units: list[int] = field(default_factory=lambda: [128])
    dropout_rate: float = 0.4
    num_classes: Optional[int] = None
    input_shape: Optional[tuple[int, int, int]] = None


@dataclass
class TrainingConfig:
    """Parámetros de entrenamiento."""

    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    loss_function: str = "categorical_crossentropy"
    # Métrica principal: F1-macro (ver documentación arriba)
    primary_metric: str = "f1_macro"
    secondary_metric: str = "accuracy"
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    # ReduceLROnPlateau
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 5
    lr_min: float = 1e-6
    # Validación cruzada
    k_folds: int = 5
    random_seed: int = 42


@dataclass
class PathsConfig:
    """Rutas del proyecto. Modifica BASE_DIR si cambias de máquina."""

    base_dir: Path = field(default_factory=lambda: Path("./data"))
    raw_audio_dir: str = "raw"
    processed_dir: str = "processed"
    spectrograms_dir: str = "spectrograms"
    models_dir: str = "models"
    results_dir: str = "results"
    logs_dir: str = "logs"

    @property
    def raw_audio(self) -> Path:
        return self.base_dir / self.raw_audio_dir

    @property
    def processed(self) -> Path:
        return self.base_dir / self.processed_dir

    @property
    def spectrograms(self) -> Path:
        return self.base_dir / self.spectrograms_dir

    @property
    def models(self) -> Path:
        return self.base_dir / self.models_dir

    @property
    def results(self) -> Path:
        return self.base_dir / self.results_dir

    @property
    def logs(self) -> Path:
        return self.base_dir / self.logs_dir

    def create_all(self) -> None:
        """Crea todos los directorios necesarios."""
        for prop_name in ["raw_audio", "processed", "spectrograms",
                          "models", "results", "logs"]:
            getattr(self, prop_name).mkdir(parents=True, exist_ok=True)


@dataclass
class ProjectConfig:
    """Configuración maestra del proyecto.

    Uso:
        config = ProjectConfig()
        config.paths.create_all()

        # Para cambiar algo:
        config.audio.sample_rate = 44100
        config.training.batch_size = 64
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    crnn: CRNNConfig = field(default_factory=CRNNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    # Metadata del dataset (se llena al cargar datos)
    genre_labels: list[str] = field(default_factory=list)

    def setup_input_shapes(self) -> None:
        """Calcula las dimensiones de entrada basándose en AudioConfig."""
        height = self.audio.n_mels
        width = self.audio.spectrogram_width
        channels = 1  # Espectrograma en escala de grises
        shape = (height, width, channels)
        self.model.input_shape = shape
        self.crnn.input_shape = shape

    def setup_num_classes(self, num_classes: int) -> None:
        """Configura el número de clases en ambos modelos."""
        self.model.num_classes = num_classes
        self.crnn.num_classes = num_classes
