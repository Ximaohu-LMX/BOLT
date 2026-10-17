"""Column-cardinality adapter. PostgreSQL slot extraction is a separate stage."""
import math
from .protocol import require


def selectivity(estimate, total, nonnull, negated=False):
    require(type(total) is int and type(nonnull) is int and 0 <= nonnull <= total and total > 0,
            'invalid column row counts')
    estimate = float(estimate)
    require(math.isfinite(estimate) and 0 <= estimate <= nonnull, 'prediction outside non-NULL domain')
    return (nonnull-estimate if negated else estimate)/total


def positive_predictions(patterns, predictions, metadata):
    """Require exactly the neural targets; empty pattern uses a common summary."""
    expected = set(patterns)
    require(set(predictions) == expected-{''}, 'missing/unexpected neural test predictions')
    result = dict(predictions)
    if '' in expected:
        result[''] = metadata['empty_strings']
    for pred in result.values():
        selectivity(pred, metadata['rows'], metadata['nonnull'])
    return result
