# Large Dataset Information

## Generated Files

The training dataset generator creates several files:

### Included in Git Repository:
- `sample_training_challenges.json` (127KB) - Sample of 100 challenges for testing
- `standalone_training_generator.py` - Standalone generator script
- `src/advanced_training_generator.py` - Advanced generator with dependencies
- `src/generate_training_pairs.py` - Basic generator
- `TRAINING_DATASET_SUMMARY.md` - Comprehensive documentation
- `.cursorrules` - Project rules and lessons learned

### Excluded from Git (Due to Size):
- `large_training_challenges.json` (592MB) - Complete dataset of 20,000 challenges

## How to Generate the Full Dataset

To generate the complete 20,000 challenge dataset:

```bash
cd /workspace
python3 standalone_training_generator.py
```

This will create `large_training_challenges.json` with:
- 20,000 training challenges
- Grid sizes from 2x2 to 24x24
- 5 transformation categories
- 3 complexity levels
- Quality validation

## Alternative Storage Options

For the large dataset, consider:
1. **Git LFS**: Use Git Large File Storage for version control
2. **Cloud Storage**: Upload to AWS S3, Google Drive, or similar
3. **Local Generation**: Generate on-demand using the provided script
4. **Compression**: Compress the JSON file to reduce size

## Dataset Statistics

- **Total Challenges**: 20,000
- **File Size**: 592 MB (uncompressed)
- **Grid Sizes**: 23 different sizes (2x2 to 24x24)
- **Transformation Types**: 5 categories with multiple sub-types
- **Complexity Levels**: 3 tiers (Simple/Medium/Complex)
- **Quality**: All challenges validated for consistency

The generator script is fully self-contained and can recreate the entire dataset on any system with Python 3.