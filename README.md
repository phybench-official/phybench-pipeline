# api_caller TODO:
- print -> logger
- prompt configurable in config.ini

# Grading

## Usage Examples

### Command Line Interface (RECOMMENDED)
```powershell
# Use CLI with custom parameters
python -m scripts.grading --gt-file test_gt.json --gen-file test_gen.json --output-dir results.json --scoring-params "80,120" --log-file custom.log

# Use with config defaults
python -m scripts.grading

# Show help
python -m scripts.grading --help
```

### Legacy/Programmatic Usage
```python
# Import and use directly
from scripts.grading import main

# Use with config defaults
result = main('gt.json', 'gen.json', 'out.json', [60, 100], 'log.txt')
```

### Configuration Management
```ini
[GRADING]
DEFAULT_GT_FILE = ./test_gt.json
DEFAULT_GEN_FILE = ./test_gen.json
DEFAULT_OUTPUT_DIR = ./results.json
DEFAULT_SCORING_PARAMS = 60,100
NUM_PROCESSES = 0
LOG_FILE = grading.log
```
