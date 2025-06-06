# api_caller TODO:
- print -> logger
- prompt configurable in config.ini

# Grading

## Usage Examples

### Command Line Interface (RECOMMENDED)
```powershell
# Use CLI with custom parameters
python -m scripts.evaluation --gt-file test_gt.json --model-answers-file test_gen.json --output-dir results.json --scoring-params "80,120" --log-file custom.log

# Use with config defaults
python -m scripts.evaluation

# Show help
python -m scripts.evaluation --help
```

### Legacy/Programmatic Usage
```python
# Import and use directly
from scripts.evaluation import main

# Use with config defaults
result = main('gt.json', 'model_answers.json', 'out.json', [60, 100], 'log.txt')
```

### Configuration Management
```ini
[evaluation.paths]
gt_file = ./test_gt.json
model_answers_file = ./test_gen.json
output_dir = ./results.json
log_file = evaluation.log

[evaluation.scoring]
initial_score = 60
scoring_slope = 100

[evaluation.execution]
num_processes = 0
```
