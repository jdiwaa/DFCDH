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



## Usage

### Training
```bash
# Train and evaluate on Weather dataset
bash ./scripts/multivariate_forecasting/Weather/DFCDH.sh

# Train and evaluate on ECL dataset
bash ./scripts/multivariate_forecasting/ECL/DFCDH.sh

# Train and evaluate on ETTm1 dataset
bash ./scripts/multivariate_forecasting/DFCDH_ETTm1.sh

# Train and evaluate on ETTm2 dataset
bash ./scripts/multivariate_forecasting/DFCDH_ETTm2.sh

# Train and evaluate on ETTh1 dataset
bash ./scripts/multivariate_forecasting/DFCDH_ETTh1.sh

# Train and evaluate on ETTh2 dataset
bash ./scripts/multivariate_forecasting/DFCDH_ETTh2.sh
```

# Key Model Components

Our DFCDH model consists of the following major modules, illustrated in Figure 3 of the paper:

### 1. Frequency-enhanced Embedding Layer
- Processes input data in the frequency domain to capture important periodic patterns.
- Embeds the frequency-processed data into high-dimensional tokens for richer representations.

### 2. Mask Matrix Generator
- Analyzes differences in seasonal and trend components among variables.
- Generates mask matrices that guide the channel attention mechanism by controlling inter-channel interactions.

### 3. Masked Channel Attention Layer
- Captures dependencies across different channels to effectively fuse multi-channel information.
- Uses the mask matrices to prevent fusion between unrelated channels, enhancing robustness against data heterogeneity.

### 4. Gated Linear Unit (GLU) Layer
- Performs nonlinear transformations on features at each position to extract complex feature interactions.

### 5. Projection Layer
- Transforms the feature dimensions to produce the final prediction outputs.

---

