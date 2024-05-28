APP=/software/containers/versions/rf_diffusion_aa/24-05-21/rf_diffusion_aa.sif
PYTHONPATH=.. $APP -mpytest --benchmark-skip --ignore tests/test_semantics.py --durations=10 tests


