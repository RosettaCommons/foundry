"""Validation for sequential RF3 jobs; deliberately independent of model loading."""

from pathlib import Path


def validate_jobs(jobs):
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a nonempty list of inputs/out_dir/seed objects")
    outputs = set()
    validated = []
    for job in jobs:
        if not isinstance(job, dict) or set(job) != {"inputs", "out_dir", "seed"}:
            raise ValueError("Each job must contain exactly inputs, out_dir and seed")
        if any(
            not isinstance(job[k], str) or not job[k].strip()
            for k in ("inputs", "out_dir")
        ):
            raise ValueError("Job inputs and out_dir must be nonempty path strings")
        if type(job["seed"]) is not int or not 0 <= job["seed"] < 2**32:
            raise ValueError("Job seed must be an integer in [0, 2**32)")
        output = Path(job["out_dir"]).resolve()
        if output in outputs:
            raise ValueError("Jobs must have distinct output directories")
        outputs.add(output)
        validated.append(dict(job))
    return validated
