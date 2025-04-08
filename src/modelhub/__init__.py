import logging

from beartype.claw import beartype_this_package
from environs import Env
from jaxtyping import install_import_hook

# Load environment variables from `.env` file
_env = Env()
_env.read_env()
should_typecheck = _env.bool("TYPE_CHECK", default=False)
should_debug = _env.bool("DEBUG", default=False)
should_check_nans = _env.bool("NAN_CHECK", default=True)

# Set up logger
logger = logging.getLogger("modelhub")
# ... set logging level based on `DEBUG` environment variable
logger.setLevel(logging.DEBUG if should_debug else logging.INFO)
# ... log the current mode
logger.debug("Debug mode: %s", should_debug)
logger.debug("Type checking mode: %s", should_typecheck)
logger.debug("NAN checking mode: %s", should_check_nans)

# Enable runtime type checking if `TYPE_CHECK` environment variable is set to `True`
if should_typecheck:
    beartype_this_package()
    install_import_hook("modelhub", "beartype.beartype")
