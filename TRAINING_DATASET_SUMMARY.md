# Large-Scale ARC-AGI Training Dataset

## Overview

Successfully generated **20,000 training challenges** for ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence) covering grid sizes from 2x2 to 24x24. This dataset provides a comprehensive training resource for developing AI systems capable of abstract reasoning and pattern recognition.

## Dataset Statistics

### Total Challenges: 20,000
- **File Size**: 592 MB (592,111,619 bytes)
- **Format**: JSON with ARC-AGI compatible structure
- **Location**: `/workspace/large_training_challenges.json`

### Grid Size Distribution
| Grid Size | Challenges | Percentage |
|-----------|------------|------------|
| 2x2       | 297        | 1.5%       |
| 3x3       | 661        | 3.3%       |
| 4x4       | 814        | 4.1%       |
| 5x5       | 897        | 4.5%       |
| 6x6       | 905        | 4.5%       |
| 7x7       | 904        | 4.5%       |
| 8x8       | 906        | 4.5%       |
| 9x9       | 905        | 4.5%       |
| 10x10     | 913        | 4.6%       |
| 11x11     | 915        | 4.6%       |
| 12x12     | 906        | 4.5%       |
| 13x13     | 916        | 4.6%       |
| 14x14     | 908        | 4.5%       |
| 15x15     | 925        | 4.6%       |
| 16x16     | 905        | 4.5%       |
| 17x17     | 923        | 4.6%       |
| 18x18     | 927        | 4.6%       |
| 19x19     | 914        | 4.6%       |
| 20x20     | 919        | 4.6%       |
| 21x21     | 917        | 4.6%       |
| 22x22     | 906        | 4.5%       |
| 23x23     | 907        | 4.5%       |
| 24x24     | 910        | 4.6%       |

### Challenge Type Distribution
| Type | Challenges | Percentage | Description |
|------|------------|------------|-------------|
| **Color Operations** | 6,684 | 33.4% | Color replacement, inversion, gradients |
| **Shape Operations** | 7,365 | 36.8% | Rotation, flipping, scaling, movement |
| **Pattern Recognition** | 2,630 | 13.2% | Edge detection, object counting, symmetry |
| **Mathematical** | 1,976 | 9.9% | Neighbor sums, position-based operations |
| **Geometric** | 1,345 | 6.7% | Distance calculations, angle computations |

### Complexity Distribution
| Complexity | Challenges | Percentage | Description |
|------------|------------|------------|-------------|
| **Simple** | 11,678 | 58.4% | Basic transformations, single-step operations |
| **Medium** | 5,363 | 26.8% | Multi-step operations, conditional logic |
| **Complex** | 2,959 | 14.8% | Advanced patterns, hierarchical operations |

## Challenge Structure

Each challenge follows the ARC-AGI format:

```json
{
  "challenge_id": {
    "id": "type_gridsize_complexity_randomid",
    "train": [
      {
        "input": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        "output": [[1, 2, 3], [2, 1, 2], [3, 2, 1]]
      }
    ],
    "test": [
      {
        "input": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
        "output": [[1, 2, 3], [2, 1, 2], [3, 2, 1]]
      }
    ]
  }
}
```

## Transformation Categories

### 1. Color Operations (33.4%)
- **Simple**: Color replacement, inversion, shifting
- **Medium**: Conditional replacement, color gradients
- **Complex**: Multi-color transformations, color sequences

### 2. Shape Operations (36.8%)
- **Simple**: 90° rotation, horizontal/vertical flipping
- **Medium**: 2x scaling, shape movement, extraction
- **Complex**: Complex rotations, shape morphing

### 3. Pattern Recognition (13.2%)
- **Simple**: Edge detection, object counting, symmetry detection
- **Medium**: Pattern matching, feature extraction
- **Complex**: Hierarchical patterns, multi-pattern recognition

### 4. Mathematical Operations (9.9%)
- **Simple**: Neighbor sums, position-based multiplication
- **Medium**: Gradient calculations, distance transforms
- **Complex**: Multi-operation sequences, recursive operations

### 5. Geometric Operations (6.7%)
- **Simple**: Distance from center, angle calculations
- **Medium**: Voronoi diagrams, Delaunay triangulation
- **Complex**: Hierarchical geometric operations

## Quality Assurance

- **Validation**: Each challenge undergoes quality validation
- **Consistency**: Input/output dimensions match across all examples
- **Diversity**: Ensures color variation and non-trivial transformations
- **Error Handling**: Robust generation with fallback mechanisms

## Usage

### Loading the Dataset
```python
import json

# Load the full dataset
with open('large_training_challenges.json', 'r') as f:
    challenges = json.load(f)

# Load a sample (100 challenges)
with open('sample_training_challenges.json', 'r') as f:
    sample_challenges = json.load(f)
```

### Integration with Existing ARC-AGI Systems
The generated challenges are fully compatible with existing ARC-AGI evaluation frameworks and can be used directly with:
- `src.models.Challenge` and `src.models.Example` classes
- Existing evaluation pipelines in `src.logic.py`
- Visualization tools in `src.reps.py`

## Files Generated

1. **`large_training_challenges.json`** (592 MB) - Complete dataset of 20,000 challenges
2. **`sample_training_challenges.json`** (3 MB) - Sample of 100 challenges for testing
3. **`standalone_training_generator.py`** - Standalone generator script
4. **`TRAINING_DATASET_SUMMARY.md`** - This documentation

## Technical Implementation

The generator uses a sophisticated approach with:
- **Modular Design**: Separate generators for each transformation type
- **Complexity Levels**: Three-tier complexity system (Simple/Medium/Complex)
- **Quality Control**: Built-in validation and error handling
- **Scalability**: Efficient generation for large datasets
- **Flexibility**: Configurable parameters for different use cases

## Next Steps

This dataset provides a solid foundation for:
1. **Training ARC-AGI models** with diverse, high-quality examples
2. **Benchmarking** different AI approaches to abstract reasoning
3. **Research** into pattern recognition and transformation learning
4. **Development** of more sophisticated reasoning systems

The dataset covers the full range of grid sizes from 2x2 to 24x24 as requested, providing thousands of training pairs across all dimensions for comprehensive model training.