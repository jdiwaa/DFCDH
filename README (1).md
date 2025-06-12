# Beyond Data Heterogeneity: A Multivariate Time Series Forecastingfor Energy Systems through Enhanced Channel Fusion in Frequency Domain

**Official Implementation**

## Installation

### Requirements
```bash
Python >= 3.10
PyTorch >= 2.4.1
CUDA >= 12.1 (recommended for GPU acceleration)
```

### Dependencies
```bash
pip install torch torchvision 
pip install scikit-learn matplotlib
pip install numpy pandas 
```

## Project Structure

```
├── model/
│   ├── __init__.py                 # Model module exports
│   ├── DFCDH.py                    # Main DFCDH model implementation
│       ├── moving_avg              # Moving average block to highlight the trend of time series
│       ├── series_decomp           # Series decomposition block
│       ├── Mahalanobis_mask        # Mask generator using Mahalanobis-style distance for dynamic channel selection
│       └── Model                   # Core DFCDH forecasting model with frequency-enhanced embedding and masked attention
├── layers/
│   ├── __init__.py                 # Layer module exports
│   ├── Embed.py                    # Embedding modules for input values and time features
│   │   ├── PositionalEmbedding     # Fixed positional encoding using sine and cosine functions
│   │   ├── TokenEmbedding          # Conv1D-based token embedding for input sequences
│   │   ├── FixedEmbedding          # Fixed embedding layer for time features
│   │   ├── TemporalEmbedding       # Temporal feature embedding using discrete encodings (month, day, hour...)
│   │   ├── TimeFeatureEmbedding    # Alternative time encoding via linear projection (continuous time features)
│   │   ├── DataEmbedding           # Combined embedding: value + positional + temporal
│   │   └── DataEmbedding_inverted  # Inverted variant with optional covariate (e.g., timestamp) fusion
│   ├── SelfAttention_Family.py     # Self-attention related modules
│   │   ├── FullAttention           # Standard scaled dot-product attention
│   │   └── AttentionLayer          # Multi-head attention
│   └── Transformer_EncDec.py       # Transformer encoder-decoder layers
│       ├── ConvLayer               # 1D convolution + normalization + pooling
│       ├── EncoderLayer            # Transformer encoder block with attention and FFN
│       ├── Encoder                 # Stack of encoder layers with optional conv layers
│       ├── DecoderLayer            # Transformer decoder block with self- and cross-attention
│       └── Decoder                 # Stack of decoder layers with optional norm and projection
├── script/
│   └── multivariate_forecasting/
│       ├── ECL/
│       │   ├── DFCDH.sh            # shell scripts related to the ECL dataset
│       ├── ETT/
│       │   ├── DFCDH_*.sh          # Shell scripts for the ETT dataset
│       └── Weather/
│           └── DFCDH.sh            # Shell scripts for the Weather dataset
└── run.py                         # Main training and evaluation script
```

## Model Parameter Explanation

- `is_training` (int): Indicates training status; 1 for training mode, 0 for evaluation/testing.
- `model_id` (str): Identifier for the current model experiment.
- `model` (str): Model name to be used (e.g., "DFCDH").

### Data Loading
- `data` (str): Dataset type, e.g., "custom" or predefined dataset names.
- `root_path` (str): Root directory where dataset files are stored.
- `data_path` (str): Filename of the CSV data file.
- `features` (str): Forecasting task type:
  - "M": multivariate input predicts multivariate output
  - "S": univariate input predicts univariate output
  - "MS": multivariate input predicts univariate output
- `target` (str): Target feature name for univariate forecasting tasks.
- `freq` (str): Frequency of the time series data for feature encoding; options include:
  - 's' (secondly), 't' (minutely), 'h' (hourly), 'd' (daily),
  - 'b' (business days), 'w' (weekly), 'm' (monthly),
  - or detailed frequencies like '15min' or '3h'.
- `checkpoints` (str): Directory path to save model checkpoints.

### Forecasting Task Settings
- `seq_len` (int): Length of the input sequence.
- `label_len` (int): Length of the start token sequence (mostly used in some Transformer variants; optional here).
- `pred_len` (int): Length of the prediction horizon (output sequence length).

### Model Architecture Parameters
- `enc_in` (int): Number of input features to the encoder.
- `dec_in` (int): Number of input features to the decoder.
- `c_out` (int): Number of output features predicted by the model.
- `d_model` (int): Dimension of the embedding space used throughout the model.
- `n_heads` (int): Number of attention heads in multi-head attention.
- `e_layers` (int): Number of encoder layers.
- `d_layers` (int): Number of decoder layers.
- `d_ff` (int): Dimension of the feed-forward network inside Transformer layers.
- `moving_avg` (int): Window size used for moving average smoothing (for trend extraction).
- `factor` (int): Factor controlling attention sparsity or scaling (model-specific).
- `distil` (bool): Whether to use distillation in the encoder to reduce sequence length.
- `dropout` (float): Dropout rate applied to layers for regularization.
- `embed` (str): Method of time feature encoding, options include:
  - "timeF": time features,
  - "fixed": fixed positional encoding,
  - "learned": learned positional embedding.
- `activation` (str): Activation function used inside layers, e.g., "relu" or "gelu".
- `output_attention` (bool): Whether to output attention weights from the encoder.
- `do_predict` (bool): Whether to run prediction on unseen future data after training/testing.

### Optimization and Training Settings
- `num_workers` (int): Number of subprocesses for data loading.
- `itr` (int): Number of experiment repetitions.
- `train_epochs` (int): Number of training epochs.
- `batch_size` (int): Mini-batch size for training.
- `patience` (int): Patience for early stopping.
- `learning_rate` (float): Learning rate for the optimizer.
- `des` (str): Experiment description or notes.
- `loss` (str): Loss function used for training, e.g., "MSE".
- `lradj` (str): Learning rate adjustment strategy.
- `use_amp` (bool): Whether to use Automatic Mixed Precision training for speedup.

### Hardware and Device Settings
- `use_gpu` (bool): Whether to use GPU acceleration.
- `gpu` (int): Default GPU device index.
- `use_multi_gpu` (bool): Whether to use multiple GPUs.
- `devices` (str): Comma-separated string listing GPU device IDs.

### Additional Flags
- `exp_name` (str): Experiment name.
- `channel_independence` (bool): Whether to use channel independence mechanism.
- `inverse` (bool): Whether to inverse the output data.
- `class_strategy` (str): Classification strategy in the model (e.g., 'projection', 'average', 'cls_token').
- `target_root_path` (str): Root path for target data file.
- `target_data_path` (str): Target data filename.
- `efficient_training` (bool): Whether to enable efficient training mode (partial training).
- `use_norm` (bool): Whether to apply normalization and denormalization.
- `partial_start_index` (int): Start index of variates for partial training in multi-variate data.


## Usage

### Basic Training
```bash
# Train on Chengdu as target city
python main.py --config config/chengdu_config.yaml

# Train on METR-LA as target city
python main.py --config config/metr-la_config.yaml

# Train on PEMS-BAY as target city
python main.py --config config/pems-bay_config.yaml

# Train on Shenzhen as target city
python main.py --config config/shenzhen_config.yaml
```



### Custom Configuration
```bash
# Override configuration parameters
python main.py --config config/chengdu_config.yaml \
    --seed 42 \
    --cuda 0 \
    --seq_len 24 \
    --horizons 3，6，12
```

## Key Model Components

### 1. OptimizedTASA (Main Model)
- **Semantic-Topological Decoupling**: Separates shared traffic semantics from city-specific topology
- **Parameter Separation**: Distinguishes between shared, private, and domain-specific parameters

### 2. Spatio-Temporal Adaptive Embedding (STAE)
- **Multi-Modal Embedding**: Processes traffic data, temporal patterns, and spatial relationships
- **Temporal Features**: Automatic generation of time-of-day and day-of-week embeddings
- **Spatial Features**: Node-specific and adaptive spatial embeddings

### 3. Semi-Self Attention Mechanism
- **Parameter Separation**: Uses shared parameters for Q,K and private parameters for V
- **LoRA Adaptation**: Efficient low-rank adaptation for city-specific fine-tuning
- **Attention Variants**: Standard, Flash, and Local attention implementations

### 4. Spatial Transformer
- **Dual Attention**: Intra-block and inter-block attention for hierarchical spatial modeling
- **Block Partitioning**: KD-tree based spatial partitioning for scalability
- **Efficient Implementation**: LoRA-enhanced MLPs for reduced parameter overhead
