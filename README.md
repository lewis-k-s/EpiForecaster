# Dual Graph Neural Network for Epidemiological Forecasting

A PyTorch Geometric implementation for epidemiological forecasting using a novel dual graph architecture that combines mobility flow networks with wastewater treatment plant (EDAR) surveillance signals for enhanced disease prediction in Catalonia, Spain.

## Overview

This project implements a sophisticated dual graph neural network framework for epidemiological forecasting, leveraging both human mobility patterns and wastewater biomarker signals. The system processes origin-destination (O-D) mobility flows from NetCDF data and integrates them with EDAR (wastewater treatment plant) monitoring data to predict disease spread patterns across Catalan municipalities.

### Key Features

- **Dual Graph Architecture**: Separate GraphSAGE encoders for mobility and EDAR networks with attention-based fusion
- **Multi-Modal Data Integration**: NetCDF mobility flows, CSV population data, and wastewater biomarker signals
- **Attention Masking**: Uses real-world EDAR-municipality contribution ratios as domain-informed attention weights
- **Temporal Forecasting**: LSTM-based temporal encoding with configurable forecast horizons
- **Inductive Learning**: Generalizes to unseen municipalities using GraphSAGE-inspired architecture
- **Streaming Data Support**: Efficient processing of large temporal NetCDF datasets with dask chunking

## Architecture

### Dual Graph Structure

The system constructs two separate but interconnected graphs:

1. **Mobility Graph**:
   - **Nodes**: Catalan municipalities with epidemiological case data
   - **Edges**: Origin-destination mobility flows with flow volumes as edge attributes
   - **Features**: Population data, geographic coordinates, and flow-derived statistics

2. **EDAR Graph**:
   - **Nodes**: Wastewater treatment plants with biomarker measurements
   - **Connections**: Attention masks based on municipality-EDAR contribution ratios
   - **Features**: Wastewater signal data and treatment plant characteristics

### Core Components

```
├── data/
│   ├── mobility_loader.py        # NetCDF mobility data processing with streaming support
│   ├── edar_attention_loader.py  # EDAR contribution ratio processing
│   └── feature_extractor.py      # Geometric and spatial feature extraction
├── models/
│   ├── graphsage_od.py          # GraphSAGE for origin-destination mobility data
│   ├── dual_graph_sage.py       # Dual graph architecture with attention fusion
│   ├── aggregators.py           # Multiple aggregation strategies (mean, attention, maxpool)
│   ├── attention_mask.py        # Multi-scale attention masking for EDAR integration
│   └── dual_graph_forecaster.py # Complete forecasting pipeline with LSTM temporal encoding
├── graph/
│   ├── node_encoder.py          # Inductive node feature encoding
│   └── edge_processor.py        # Edge feature processing with outlier detection
└── main.py                      # Training pipeline with comprehensive evaluation
```

## Methodology

### Dual Graph Neural Network Architecture

The system implements a novel dual graph architecture that processes mobility and wastewater signals separately before fusing them through attention mechanisms:

```python
class DualGraphSAGE(nn.Module):
    def __init__(self, mobility_input_dim, edar_input_dim, hidden_dim=128):
        super().__init__()
        # Separate encoders for each graph
        self.mobility_encoder = GraphSAGE_OD(
            input_dim=mobility_input_dim, 
            hidden_dim=hidden_dim, 
            num_layers=2
        )
        self.edar_encoder = GraphSAGE_OD(
            input_dim=edar_input_dim, 
            hidden_dim=hidden_dim//2, 
            num_layers=2
        )
        self.attention_processor = AttentionMaskProcessor(
            hidden_dim, contribution_ratios
        )
```

### Origin-Destination GraphSAGE

Specialized GraphSAGE implementation for mobility flow data:

```python
class GraphSAGE_OD(nn.Module):
    """GraphSAGE optimized for origin-destination flow patterns"""
    def __init__(self, input_dim, hidden_dim, aggregator_type='mean'):
        super().__init__()
        self.aggregator = create_aggregator(aggregator_type, hidden_dim)
        self.sage_convs = nn.ModuleList([
            SAGEConv(input_dim, hidden_dim, aggregator=self.aggregator),
            SAGEConv(hidden_dim, hidden_dim, aggregator=self.aggregator)
        ])
```

### Attention-Based Signal Fusion

The system uses real-world EDAR-municipality contribution ratios as attention weights:

```python
class AttentionMaskProcessor(nn.Module):
    def __init__(self, hidden_dim, contribution_ratios):
        super().__init__()
        self.contribution_mask = torch.tensor(contribution_ratios)  # Domain knowledge
        self.learnable_attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
        
    def forward(self, mobility_features, edar_features):
        # Combine domain knowledge with learned attention
        attention_weights = self.contribution_mask * self.learnable_attention(...)
        return self.fuse_signals(mobility_features, edar_features, attention_weights)
```

### Multi-Strategy Neighborhood Aggregation

The system implements multiple aggregation strategies optimized for epidemiological transmission patterns:

- **Mean Aggregator**: Uniform importance baseline for neighborhood aggregation
- **Attention Aggregator**: Multi-head attention with edge attributes (flow volumes) as importance weights
- **Max Pool Aggregator**: Captures peak transmission risk from high-flow mobility connections
- **Edge Attribute Integration**: All aggregators incorporate mobility flow volumes and geographic distances

### Temporal Forecasting Pipeline

Complete forecasting pipeline with multi-step prediction capabilities:

```python
class DualGraphForecaster(nn.Module):
    def __init__(self, mobility_dim, edar_dim, forecast_horizon=7):
        super().__init__()
        self.dual_graph_encoder = DualGraphSAGE(mobility_dim, edar_dim)
        self.temporal_encoder = nn.LSTM(
            input_size=combined_dim, 
            hidden_size=64, 
            num_layers=1, 
            batch_first=True
        )
        # Multiple prediction heads
        self.case_count_predictor = nn.Linear(64, forecast_horizon)
        self.case_rate_predictor = nn.Sequential(
            nn.Linear(64, forecast_horizon),
            nn.Sigmoid()  # For rate predictions [0,1]
        )
        
    def forward(self, mobility_graphs, edar_data, temporal_sequence):
        # Process dual graphs
        spatial_embeds = self.dual_graph_encoder(mobility_graphs, edar_data)
        # Temporal modeling
        temporal_embeds, _ = self.temporal_encoder(spatial_embeds)
        # Multi-target predictions
        return {
            'case_counts': self.case_count_predictor(temporal_embeds),
            'case_rates': self.case_rate_predictor(temporal_embeds)
        }
```

## Implementation Details

### Data Processing Pipeline

The system processes multiple data modalities with specialized loaders:

```python
from data.mobility_loader import MobilityDataLoader
from data.edar_attention_loader import create_edar_attention_loader
from data.feature_extractor import GeometricFeatureExtractor

# NetCDF mobility data with streaming support
mobility_loader = MobilityDataLoader(
    normalize_flows=True,
    min_flow_threshold=10,
    engine='h5netcdf',
    chunks={'time': 1}
)

# EDAR attention masks from contribution ratios
edar_loader = create_edar_attention_loader(
    edar_edges_path='data/files/edar_muni_edges.nc',
    normalize_contributions=True
)

# Geographic feature extraction
feature_extractor = GeometricFeatureExtractor(
    distance_threshold_km=50,
    k_neighbors=10
)
```

### Training Configuration

- **Optimizer**: Adam with learning rate 0.001, weight decay 1e-4
- **Loss Functions**: MSE for case counts, with multiple evaluation metrics (MAE, MAPE, RMSE, R²)
- **Regularization**: Dropout (0.2-0.5), gradient clipping (max norm 1.0)
- **Learning Rate Scheduling**: ReduceLROnPlateau with patience=10
- **Early Stopping**: Patience of 20 epochs based on validation loss
- **Hardware**: Optimized for single GPU training with CUDA support

## Data Requirements

### Input Data Formats

1. **NetCDF Mobility Data** (`.nc` files):
   - **Format**: MITMA origin-destination mobility flows
   - **Dimensions**: `time × home × destination` (e.g., 365 × 584 × 584 for Catalonia)
   - **Variables**: `person_hours(time, home, destination)` with flow volumes
   - **Coordinates**: Spanish administrative codes (INE municipality codes)

2. **CSV Population Data**:
   - **File**: `fl_population_por_municipis.csv`
   - **Columns**: `[id, d.population, d.density_pop_m2]`
   - **Purpose**: Node features for demographic context

3. **EDAR Network Data** (`.nc` files):
   - **File**: `edar_muni_edges.nc`
   - **Content**: Municipality-to-EDAR contribution ratios matrix
   - **Purpose**: Attention masking for wastewater signal integration

4. **Case Data** (CSV):
   - **File**: `flowmaps_cat_municipio_cases.csv`
   - **Content**: Time-series epidemiological case counts by municipality

5. **Wastewater Biomarkers** (CSV):
   - **File**: `wastewater_biomarkers_icra.csv`
   - **Content**: EDAR-level wastewater surveillance measurements

### Data Directory Structure

```
data/files/
├── daily_dynpop_mitma/          # NetCDF mobility files by date
├── edar_muni_edges.nc           # EDAR-municipality contribution ratios
├── fl_population_por_municipis.csv
├── flowmaps_cat_municipio_cases.csv
├── wastewater_biomarkers_icra.csv
└── geo/
    ├── EDAR.geojson            # EDAR facility locations
    └── fl_municipios_catalonia.geojson  # Municipality boundaries
```

### Preprocessing Pipeline

```python
from data.mobility_loader import MobilityDataLoader

# Load and preprocess mobility data
mobility_loader = MobilityDataLoader(
    normalize_flows=True,
    min_flow_threshold=10,
    include_self_loops=False
)

# Create dataset with preprocessing hooks
dataset = mobility_loader.create_dataset(
    netcdf_filepath='data/files/daily_dynpop_mitma/2023-03-15.nc',
    population_filepath='data/files/fl_population_por_municipis.csv'
)

# Stream temporal sequences for training
temporal_graphs = list(mobility_loader.stream_dataset(
    netcdf_filepath=dataset["netcdf_filepath"],
    time_slice=slice(0, 50)  # Process first 50 days
))
```

## Installation

```bash
# Using uv (recommended for this project)
uv sync

# Install development dependencies
uv sync --group dev

# Or manually install with pip
pip install -e .
```

### Key Dependencies

- **PyTorch Ecosystem**: `torch>=2.0.0`, `torch-geometric>=2.3.0`
- **NetCDF Processing**: `xarray>=2024.7.0`, `netcdf4>=1.7.2`, `h5netcdf>=1.6.4`
- **Scientific Computing**: `numpy>=1.24.0`, `pandas>=2.0.0`, `dask>=2024.8.0`
- **Geospatial**: `geopandas>=0.13.0`, `shapely>=2.0.0`

## Usage

### Basic Training

```bash
# Train with default configuration (uses directory: files/daily_dynpop_mitma/)
uv run python main.py

# Enable preprocessing hooks for data cleaning
uv run python main.py --enable_preprocessing_hooks

# Quick validation with single file (reduced epochs)
uv run python main.py --mobility files/daily_dynpop_mitma/mitma_mov_cat.daily_personhours.2020-02-01_2020-02-29.nc --epochs 5
```

### Data Selection and Filtering

```bash
# Single NetCDF file
uv run python main.py --mobility files/daily_dynpop_mitma/mitma_mov_cat.daily_personhours.2020-02-01_2020-02-29.nc

# Directory with date range filtering
uv run python main.py --mobility files/daily_dynpop_mitma/ --start-date 2020-02-01 --end-date 2020-02-29

# Multiple months of data
uv run python main.py --mobility files/daily_dynpop_mitma/ --start-date 2020-02-01 --end-date 2020-04-30 --epochs 50
```

### Advanced Configuration

```bash
# Use specific aggregation strategy
uv run python main.py --aggregator attention --hidden_dim 128

# Adjust temporal and model parameters
uv run python main.py --forecast_horizon 7 --hidden_dim 256

# Full configuration with data filtering
uv run python main.py --mobility files/daily_dynpop_mitma/ --start-date 2020-02-01 --end-date 2020-03-31 --aggregator attention --hidden_dim 256 --epochs 100
```

### CLI Arguments

- `--mobility`: Path to NetCDF file or directory (default: `files/daily_dynpop_mitma/`)
- `--start-date`: Start date filter in YYYY-MM-DD format (directory mode only)
- `--end-date`: End date filter in YYYY-MM-DD format (directory mode only)  
- `--aggregator`: Aggregation strategy: `mean`, `attention`, `max`, `lstm`, `hybrid` (default: `attention`)
- `--epochs`: Number of training epochs (default: 10)
- `--hidden_dim`: Model hidden dimension (default: 128)
- `--forecast_horizon`: Days to forecast (default: 7)

### Python API Usage

```python
from data.mobility_loader import MobilityDataLoader
from data.edar_attention_loader import create_edar_attention_loader
from models.dual_graph_forecaster import create_dual_graph_forecaster

# Setup data loaders
mobility_loader = MobilityDataLoader(normalize_flows=True, min_flow_threshold=10)
edar_loader = create_edar_attention_loader('data/files/edar_muni_edges.nc')

# Create dual graph forecaster
forecaster = create_dual_graph_forecaster(
    mobility_input_dim=64,
    edar_input_dim=32, 
    hidden_dim=128,
    forecast_horizon=7
)

# Load temporal graph sequence
temporal_graphs = list(mobility_loader.stream_dataset(
    netcdf_filepath='data/files/daily_dynpop_mitma/2023-03-15.nc',
    time_slice=slice(0, 30)
))
```

### Model Evaluation

The training pipeline automatically computes comprehensive metrics:

- **MSE** (Mean Squared Error) - primary training loss
- **MAE** (Mean Absolute Error) - robust error measurement
- **MAPE** (Mean Absolute Percentage Error) - relative error assessment  
- **RMSE** (Root Mean Squared Error) - penalty for large errors
- **R²** (Coefficient of Determination) - explained variance

## Performance Considerations

### Computational Efficiency

- **NetCDF Streaming**: Dask-based chunking for processing large temporal datasets without memory overflow
- **Sparse Graph Processing**: Efficient edge filtering with configurable flow thresholds
- **Attention Masking**: Uses sparse tensor operations for EDAR-municipality attention computation
- **GPU Optimization**: CUDA-enabled training with automatic mixed precision support

### Scalability Features  

- **Temporal Streaming**: Processes daily mobility data incrementally for memory efficiency
- **Graph Subsampling**: Configurable neighborhood sampling for large municipality networks
- **Batch Processing**: Efficient mini-batch training with PyTorch Geometric DataLoader

### Robustness

- **Missing Data Handling**: Multiple strategies for NaN values (interpolate, zero, mean)
- **Outlier Detection**: IQR-based capping with configurable percentile thresholds  
- **Data Validation**: Comprehensive checks for coordinate validity and dimension consistency
- **Fallback Mechanisms**: SimpleDualGraphForecaster when EDAR data unavailable

## References

[1] Hamilton, W., Ying, Z., & Leskovec, J. (2017). **Inductive Representation Learning on Large Graphs**. *Advances in Neural Information Processing Systems (NeurIPS)*, 30. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)

[2] Li, Y., et al. (2024). **Learning Geospatial Region Embedding with Heterogeneous Graph**. *ACM Transactions on Knowledge Discovery from Data*, 18(5), 1-23. [doi:10.1145/3643035](https://doi.org/10.1145/3643035)

[3] Kipf, T. N., & Welling, M. (2017). **Semi-Supervised Classification with Graph Convolutional Networks**. *International Conference on Learning Representations (ICLR)*. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)

[4] Fey, M., & Lenssen, J. E. (2019). **Fast Graph Representation Learning with PyTorch Geometric**. *ICLR Workshop on Representation Learning on Graphs and Manifolds*. [arXiv:1903.02428](https://arxiv.org/abs/1903.02428)

[5] Zhang, J., et al. (2023). **Heterogeneous Graph Neural Networks for Origin-Destination Demand Prediction**. *Transportation Research Part C*, 147, 103995. [doi:10.1016/j.trc.2022.103995](https://doi.org/10.1016/j.trc.2022.103995)

## License

MIT License

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Contact

For questions and support, please open an issue on GitHub.
