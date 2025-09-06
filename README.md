# HyperSolver: Unified Framework for Combinatorial Optimization

A unified hypergraph neural network framework for solving multiple NP-hard combinatorial optimization problems using a single architecture.

## Supported Problems

- **Set Cover**: Find minimum sets to cover all elements
- **Hitting Set**: Find minimum elements to hit all sets  
- **Subset Sum**: Find subset closest to target sum
- **Hypergraph Max Cut**: Partition nodes to maximize cut hyperedges
- **Hypergraph Multiway Cut**: Partition nodes into k groups maximizing cuts

## Installation

### Quick Setup

```bash
git clone <repository-url>
cd hypersolver-repo
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Alternative: Automated Installation

```bash
./INSTALL.sh
```

The installer will prompt you to choose between:
1. **Minimal dependencies** (recommended for most users)
2. **Complete environment** (exact development versions)

## Usage

### Basic Examples

```bash
# Set Cover Problem
python run.py --problem set_cover

# Subset Sum Problem  
python run.py --problem subset_sum

# Hypergraph Max Cut
python run.py --problem hypermaxcut

# Hypergraph Multiway Cut (3-way partitioning)
python run.py --problem hypermultiwaycut

# Hitting Set Problem
python run.py --problem hitting_set
```

### Training Modes

**Instance-Specific Training** (default)
```bash
python run.py --problem set_cover --mode instance_specific
```
Trains a new model from scratch for each problem instance. Provides best performance but takes longer.

**Pretraining Mode**
```bash
python run.py --problem hypermaxcut --mode pretrain --pretrained_model_path models/set_cover.pth
```
Uses transfer learning from a pretrained model. Faster training with maintained solution quality.

**Test-Only Mode**
```bash
python run.py --problem subset_sum --mode test_only --pretrained_model_path models/subset_sum.pth
```
Evaluates using an existing trained model without additional training.

**Test with Fine-tuning**
```bash
python run.py --problem hitting_set --mode test_finetune --pretrained_model_path models/set_cover.pth
```
Loads a pretrained model and performs limited fine-tuning before evaluation.

## Configuration

Edit JSON files in the `configs/` directory to modify:
- Problem parameters (number of nodes, hyperedges)
- Training settings (learning rate, epochs, patience)
- Data paths and file locations
- Neural network hyperparameters (hidden dimensions, layers)

### Key Configuration Files
- `set_cover_config.json`: Set cover problem settings
- `subset_sum_config.json`: Subset sum problem settings  
- `hypermaxcut_config.json`: Hypergraph max cut settings
- `hypermultiwaycut_config.json`: Multiway cut settings

## Data Format

### Set Cover and Hitting Set
```
<num_elements> <num_subsets>
<element_ids_in_subset_1>
<element_ids_in_subset_2>
...
```

### Subset Sum
```
<num_items> <target_sum>
<weight_1> <weight_2> ... <weight_n>
```

### Hypergraph Problems
```
<num_nodes> <num_hyperedges>
<node_ids_in_hyperedge_1>
<node_ids_in_hyperedge_2>
...
```

## Dependencies

### Requirements Files
- **requirements.txt**: Minimal dependencies for core functionality
- **requirements_full.txt**: Complete environment with exact versions used in development

### System Requirements
- Python 3.7 or higher
- PyTorch 2.0 or higher
- NumPy 1.20 or higher
- 4GB RAM recommended
- GPU optional (CPU training supported)

## Architecture Overview

- **Unified Model**: Single neural architecture handles all problem types
- **Hypergraph Representation**: Direct modeling of multi-element constraints  
- **Unsupervised Training**: No pre-solved examples required
- **Transfer Learning**: Knowledge sharing across different problem types
- **Adaptive Training**: Automatic restart mechanisms prevent local minima

## Performance Characteristics

- **Scalability**: Linear complexity O(|V| + |E|) with problem size
- **Speed**: Up to 6.9x faster than CPLEX on large instances
- **Quality**: Near-optimal solutions across all problem types
- **Memory**: Efficient linear memory usage scaling

## Testing Installation

Run the verification script to ensure everything is working:

```bash
python test_setup.py
```

This will verify:
- All required packages are installed
- Core modules import successfully  
- Sample data files are present
- Configuration files are valid

## Troubleshooting

**Import Errors**: Ensure virtual environment is activated and dependencies installed
**Memory Issues**: Reduce problem size or use CPU-only mode for large instances
**Training Slow**: Enable GPU acceleration or reduce number of training epochs
**Configuration Errors**: Check JSON syntax in config files