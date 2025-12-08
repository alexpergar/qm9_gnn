# QM9 molecular property prediction with Graph Neural Networks

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-EE4C2C.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.7.0-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive implementation of Graph Neural Networks (GNNs) for predicting molecular properties using the QM9 dataset. This project demonstrates the progressive evolution from simple GCN architectures to state-of-the-art models like DimeNet++, showcasing the power of geometric deep learning in quantum chemistry.

## 🎯 Project overview

This project explores molecular property prediction through increasingly sophisticated GNN architectures. The primary objective is to predict heat capacity ($C_v$) values for small organic molecules from the QM9 dataset, which contains approximately 134,000 molecules with up to 9 heavy atoms (C, O, N, F).

### Key Features

- **Multiple GNN architectures**: Implementation of 4 different model architectures with increasing complexity
- **Modular design**: Clean separation of data processing, model definitions, training, and evaluation
- **Production-ready code**: Structured codebase with reusable components and comprehensive documentation
- **Interactive notebooks**: Jupyter notebooks for experimentation and visualization
- **Performance optimization**: Advanced training techniques including early stopping, learning rate scheduling, and gradient clipping

## 📊 Models Implemented

### 1. Simple GCN (Graph Convolutional Network)

A baseline implementation using basic graph convolutions with global mean pooling.

**Architecture:**

- 3 GCN layers with ReLU activation
- Hidden dimension: 128
- Global mean pooling for graph-level predictions
- Simple linear output layer

### 2. Improved GCN

Enhanced GCN with residual connections, batch normalization, and deeper architecture.

**Key Improvements:**

- 5 GCN layers with residual connections
- Batch normalization for training stability
- Two-layer MLP readout with increased dropout (0.3)
- Better gradient flow through skip connections

### 3. NNConv (Edge-featured GNN)

Leverages edge features through neural network convolutions for richer molecular representations.

**Architecture Highlights:**

- 5 NNConv layers with edge attribute networks
- Deep edge networks (3-layer MLPs) for learning edge transformations
- Combined mean and sum pooling for comprehensive graph representation
- Layer normalization and residual connections
- Hidden dimension: 256

### 4. DimeNet++ (Directional message passing)

State-of-the-art architecture using 3D molecular geometry and directional message passing.

**Advanced Features:**

- Directional message passing based on bond angles and distances
- Spherical basis functions for geometric modeling
- 4 interaction blocks with 128 hidden channels
- Radial basis functions with 5.0 Å cutoff
- Specialized for 3D molecular structures

## 🗂️ Project Structure

```
qm9/
├── data/
│   └── QM9/                    # QM9 dataset (auto-downloaded)
│       ├── raw/
│       └── processed/
├── models/                      # Trained model checkpoints
│   ├── simple_gcn.pt
│   ├── improved_gcn.pt
│   ├── nnconv_model.pt
│   └── dimenetplusplus.pt
├── notebooks/                   # Interactive Jupyter notebooks
│   ├── simple_gcn.ipynb
│   ├── improved_gcn.ipynb
│   ├── nnconv.ipynb
│   └── dimenetplusplus.ipynb
├── scripts/
│   ├── data/
│   │   └── dataset.py          # Dataset loading and preprocessing
│   ├── model/
│   │   ├── simple_gcn.py       # Simple GCN model
│   │   ├── improved_gcn.py     # Enhanced GCN with residuals
│   │   ├── nnconv.py           # Edge-featured NNConv model
│   │   └── dimenetplusplus.py  # DimeNet++ implementation
│   ├── train/
│   │   └── train.py            # Training loop
│   └── evaluate/
│       └── evaluate.py         # Model evaluation utilities
├── environment.yml             # Conda environment specification
├── LICENSE
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- CUDA-capable GPU (optional, but recommended for faster training, specially NNConv and DimeNet++ models)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/alexpergar/qm9_gnn.git
   cd qm9_gnn
   ```

2. **Create and activate the conda environment**

   ```bash
   conda env create -f environment.yml
   conda activate qm9
   ```

3. **Verify installation**
   ```bash
   python -c "import torch; import torch_geometric; print('Sucessfull instalation')"
   ```

### Quick Start

#### Using Jupyter Notebooks

```bash
jupyter lab
```

Navigate to the `notebooks/` directory and open any notebook to start experimenting with different models.

#### Using Python Scripts

```python
import torch
from torch_geometric.loader import DataLoader
from scripts.data.dataset import load_qm9_dataset, split_dataset, normalize_train_set
from scripts.model.simple_gcn import SimpleGCN
from scripts.train.train import train_one_epoch
from scripts.evaluate.evaluate import evaluate

# Load and prepare data
dataset = load_qm9_dataset(root="data/QM9")
train_dataset, test_dataset = split_dataset(dataset, training_perc=0.9)
target_idx = 11  # Heat capacity index
train_mean, train_std = normalize_train_set(train_dataset, target_idx=target_idx)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleGCN(input_channels=dataset.num_features, hidden_channels=128).to(device)

# Train model
# ...
```

## 📈 Results

Model performance on QM9 heat capacity prediction (target property index 11):

| Model        | Parameters | Test MAE | R² Score | Relative MAE (std) |
| ------------ | ---------- | -------- | -------- | ------------------ |
| Simple GCN   | ~35K       | 1.196    | 0.704    | 0.294              |
| Improved GCN | ~180K      | 0.479    | 0.941    | 0.118              |
| NNConv       | ~69M       | 0.488    | 0.976    | 0.120              |
| DimeNet++    | ~1.9M      | 0.062    | 0.999    | 0.015              |

_Note: Results will vary based on training configuration and random seeds._

### Training Features

- **Early Stopping**: Automatic termination when validation performance plateaus
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning rate adjustment
- **Gradient Clipping**: Prevents exploding gradients during training
- **Model Checkpointing**: Saves best model based on validation performance
- **Normalization**: Target property normalization for stable training

## 🔬 Technical Details

### Dataset: QM9

The QM9 dataset contains geometric, energetic, electronic, and thermodynamic properties for 134,000+ small organic molecules. Each molecule is represented as a graph where:

- **Nodes**: Atoms with features (atomic number, hybridization, etc.)
- **Edges**: Chemical bonds with features (bond type, distance, etc.)
- **Target Property**: Heat capacity at 298.15K ($C_v$)

### Training Configuration

```python
# Typical hyperparameters
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-5
max_epochs = 500
patience = 20  # for early stopping
criterion = torch.nn.L1Loss()  # MAE loss
optimizer = torch.optim.AdamW
```

### Evaluation Metrics

- **MAE (Mean Absolute Error)**: Primary metric in original units
- **Relative MAE**: MAE normalized by standard deviation
- **R² Score**: Coefficient of determination for prediction quality
- **Baseline Comparison**: Performance vs. predicting mean value

## 🛠️ Development

### Code Organization

The project follows a modular structure with clear separation of concerns:

- **`scripts/data/`**: Data loading, preprocessing, and normalization utilities
- **`scripts/model/`**: Model architecture definitions
- **`scripts/train/`**: Training loop implementations
- **`scripts/evaluate/`**: Evaluation and prediction utilities
- **`notebooks/`**: Interactive experimentation and visualization

### Extending the Project

To add a new model:

1. Create a new model file in `scripts/model/`
2. Implement the model class inheriting from `torch.nn.Module`
3. Add training logic in a new notebook or extend existing scripts
4. Update the training and evaluation functions if needed

## 📚 Resources

### Papers & References

- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212)
- [Fast and Uncertainty-Aware Directional Message Passing](https://arxiv.org/abs/2011.14115)
- [Quantum chemistry structures and properties of 134 kilo molecules](https://www.nature.com/articles/sdata201422)

### Libraries & Tools

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/): Graph neural network library
- [PyTorch](https://pytorch.org/): Deep learning framework
- [RDKit](https://www.rdkit.org/): Chemistry toolkit (used by QM9 dataset)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Alejandro PG**

- GitHub: [@alexpergar](https://github.com/alexpergar)

---

_This project demonstrates practical applications of graph neural networks in computational chemistry and showcases progressive model development from baseline to state-of-the-art architectures._
