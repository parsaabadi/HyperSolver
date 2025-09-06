# HyperSolver Setup Instructions

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run HyperSolver on different problems:

```bash
# Set Cover
python run.py --problem set_cover

# Subset Sum  
python run.py --problem subset_sum

# Hypergraph Max Cut
python run.py --problem hypermaxcut

# Hypergraph Multiway Cut
python run.py --problem hypermultiwaycut

# Hitting Set
python run.py --problem hitting_set
```

## Configuration

Edit the JSON files in the `configs/` directory to modify problem parameters.
