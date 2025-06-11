# DFCDH

The repo is the implementation for the paper: [Beyond Data Heterogeneity: A Multivariate Time Series Forecasting for Energy Systems through Enhanced Channel Fusion in Frequency Domain].

## Usage 

1. Install Pytorch and necessary dependencies.

```
pip install -r requirements.txt
```

1. The datasets can be obtained from [Google Drive](https://drive.usercontent.google.com/download?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download&authuser=0)

2. Train and evaluate the model. We provide all the above tasks under the folder ./scripts/. You can reproduce the results as the following examples:

```
# Multivariate forecasting with DFCDH
bash ./scripts/multivariate_forecasting/Weather/DFCDH.sh

```
