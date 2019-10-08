# ml-challenge


# How to evaluate
Evaluation must be done be calling `evaluate_regression` from `challenge._internal` on your prediction data.

```python3
from challenge._internal import evaluate_regression

def your_model(...):
  ...
  
y_pred = your_model(...)
metrics = evaluate_regression(y_pred)
```

# Rules

Violating any of these items will result in disqualification.

1. Use only the datasets provided, no external sources are allowed;
2. Don't change the code in `_internal`
3. `make check-files` should exit without errors (status `0`)
