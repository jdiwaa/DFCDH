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

## Configuration Files

### Model Architecture Configuration
```yaml
model:
  # Transformer Architecture
  n_layer: 6                              # Number of transformer layers
  n_head: 4                               # Number of attention heads per layer
  n_embd: 64                              # Hidden embedding dimension
  dropout: 0.1                            # Dropout rate
  bias: false                             # Whether to use bias in linear layers
  n_linear: 1                             # Number of linear layers in MLP
  
  # Input/Output Dimensions
  input_dim: 2                            # Input feature dimensions (traffic + metadata)
  output_dim: 1                           # Output dimension (traffic speed prediction)
  
  # Temporal Embeddings
  tod_embedding_dim: 8                   # Time-of-day embedding dimension
  dow_embedding_dim: 4                    # Day-of-week embedding dimension
  
  # Spatial Embeddings
  spatial_embedding_dim: 8                # Node spatial embedding dimension
  adaptive_embedding_dim: 8               # Adaptive embedding dimension
  
  # Layer Configuration
  temporal_layers: 1                      # Number of temporal attention layers
  spatial_layers: 1                       # Number of spatial attention layers
  
  # Spatial Partitioning
  blocksize: 8                            # Nodes per spatial block
  blocknum: 4                             # Number of spatial blocks
  factors: 1                              # Spatial partitioning factor
```

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
