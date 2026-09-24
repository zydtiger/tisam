"""Exercise the JSONL adapter without encoders, data, or wall-clock timing assumptions."""

import pytest
from mammoth.logging.model import Observation

from tisam.training.observability import TrainingJsonlSink


@pytest.mark.parametrize(
    "phase,completed,batch,rate,expected",
    [("train", 2, 31, 2.0, 32.0), ("train", 2, 16, 2.0, 17.0), ("validation", 2, 1, 2.0, 2.0)],
)
def test_jsonl_batch_rate_handles_accumulation_and_partial_window(
    phase, completed, batch, rate, expected
):
    records = []

    class Writer:
        def emit(self, event, **fields):
            records.append(fields)

    sink = TrainingJsonlSink(Writer())
    sink.observe(
        Observation(
            "progress",
            fields={
                "phase": phase,
                "completed": completed,
                "coordinates": {"batch": batch},
                "throughput": rate,
            },
        )
    )
    assert records[0]["throughput"] == rate
    assert records[0]["batches_per_second"] == expected
