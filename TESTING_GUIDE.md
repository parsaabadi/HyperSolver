# HyperSolver Testing Guide

This guide provides step-by-step instructions to test HyperSolver from the GitHub repository.

## Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/parsaabadi/HyperSolver.git
cd HyperSolver
./INSTALL.sh
# OR manually:
# python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python test_setup.py
```

## Data Organization

The repository organizes test data by problem type:

```
data/
├── subset_sum/          # Subset sum test instances with various difficulty levels
├── set_cover/           # Set cover hypergraph instances of different sizes
├── hypergraph_maxcut/   # Max cut instances (shared with set_cover)
├── hypergraph_multiway/ # Multiway cut instances (shared with set_cover)  
└── hitting_set/         # Hitting set instances (shared with set_cover)
```

## Testing Each Problem Type

### Set Cover Problem
```bash
# Test with sample hypergraph data
python run.py --problem set_cover

# Uses data from ./data/set_cover/ directory
# Contains hypergraph instances of different sizes
```

### Subset Sum Problem
```bash
# Test with sample subset sum data
python run.py --problem subset_sum

# Uses data from ./data/subset_sum/ directory
# Contains test instances with various difficulty levels and item counts
```

### Hypergraph Max Cut
```bash
# Test with hypergraph data
python run.py --problem hypermaxcut

# Uses data from ./data/hypergraph_maxcut/ directory  
# Contains hypergraph instances for max cut testing
```

### Hypergraph Multiway Cut
```bash
# Test with hypergraph data (3-way partitioning)
python run.py --problem hypermultiwaycut

# Uses data from ./data/hypergraph_multiway/ directory
# Contains hypergraph instances for multiway partitioning
```

### Hitting Set Problem
```bash
# Test with hypergraph data (dual of set cover)
python run.py --problem hitting_set

# Uses data from ./data/hitting_set/ directory
# Contains test instances for hitting set problems
```

## Testing Different Training Modes

### Instance-Specific Training (Default)
```bash
python run.py --problem set_cover --mode instance_specific
```

### Transfer Learning (if you have pretrained models)
```bash
python run.py --problem hypermaxcut --mode pretrain --pretrained_model_path models/set_cover.pth
```

## Expected Output

Each run should show:
1. **Data Loading**: Confirmation of data file loading
2. **Model Initialization**: Network architecture details
3. **Training Progress**: Loss values and convergence
4. **Solution Quality**: Coverage/cut metrics
5. **Final Results**: Solution size and quality metrics

## Data File Formats

### Set Cover / Hypergraph Problems
```
<num_nodes> <num_hyperedges>
<node_ids_in_hyperedge_1>
<node_ids_in_hyperedge_2>
...
```

Example:
```
2000 4000
1 45 123 456 789
2 67 234 567 890
...
```

### Subset Sum Problems
```
<num_items> <target_sum>
<weight_1> <weight_2> ... <weight_n>
```

Example:
```
30 500000000000
12345678901 23456789012 34567890123 ...
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**2. Data File Not Found**
```bash
# Check data directory exists
ls -la data/
# Verify file paths in config files
```

**3. Memory Issues**
```bash
# Use smaller data files for testing
python run.py --problem subset_sum  # Uses smaller files automatically
```

**4. Training Too Slow**
```bash
# Reduce epochs in config files or use CPU mode
# Edit configs/*.json files to reduce max_epochs
```

### Configuration Files

Located in `configs/` directory:
- `set_cover_config.json`: Set cover parameters
- `subset_sum_config.json`: Subset sum parameters  
- `hypermaxcut_config.json`: Max cut parameters
- `hypermultiwaycut_config.json`: Multiway cut parameters

## Performance Benchmarks

Expected performance on test data:
- **Set Cover**: 95%+ coverage within 2-3 minutes
- **Subset Sum**: <1% deviation within 30 seconds
- **Max Cut**: Near-optimal cuts within 1-2 minutes
- **Multiway Cut**: 99%+ cut ratio within 2-3 minutes

## Verification Commands

```bash
# Test all problem types quickly
python run.py --problem set_cover
python run.py --problem subset_sum  
python run.py --problem hypermaxcut
python run.py --problem hypermultiwaycut
python run.py --problem hitting_set

# Check repository integrity
python test_setup.py
```

This should provide comprehensive testing of the HyperSolver framework across all supported problem types.
