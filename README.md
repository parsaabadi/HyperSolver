# HyperSolver: Unified Framework for Combinatorial Optimization

A unified hypergraph neural network framework for solving multiple NP-hard combinatorial optimization problems using a single architecture.

## Supported Problems

- **Set Cover**: Find minimum sets to cover all elements
- **Hitting Set**: Find minimum elements to hit all sets  
- **Subset Sum**: Find subset closest to target sum
- **Hypergraph Max Cut**: Partition nodes to maximize cut hyperedges
- **Hypergraph Multiway Cut**: Partition nodes into k groups maximizing cuts

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd hypersolver-repo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Examples

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

### 3. Training Modes

```bash
# Instance-specific training (default)
python run.py --problem set_cover --mode instance_specific

# Transfer learning from pretrained model
python run.py --problem hypermaxcut --mode pretrain --pretrained_model_path models/set_cover.pth
```

## Configuration

Edit JSON files in `configs/` directory to modify:
- Problem parameters
- Training settings
- Data paths
- Neural network hyperparameters

## Data Format

### Set Cover / Hitting Set
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

## System Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy 1.20+
- 4GB+ RAM recommended
- GPU optional (CPU training supported)

## Architecture

- **Unified Model**: Single neural architecture for all problems
- **Hypergraph Representation**: Direct multi-element constraint modeling  
- **Unsupervised Training**: No pre-solved examples required
- **Transfer Learning**: Knowledge sharing across problem types
- **Adaptive Training**: Automatic restart mechanisms

## Performance

- **Scalability**: Linear complexity O(|V| + |E|) 
- **Speed**: 6.9× faster than CPLEX on large instances
- **Quality**: Near-optimal solutions across all problem types
- **Memory**: Efficient linear memory usage