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
