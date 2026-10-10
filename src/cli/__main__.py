"""Entry point for ``python -m src.cli``."""

import sys

from src.cli.commands import main

sys.exit(main())
