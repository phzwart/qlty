#!/usr/bin/env python

"""The setup script (minimal, metadata now in pyproject.toml)."""

from pathlib import Path

from setuptools import setup

# Read README and HISTORY for long_description
readme_file = Path(__file__).parent / "README.rst"
history_file = Path(__file__).parent / "HISTORY.rst"

long_description = readme_file.read_text(encoding="utf-8")
if history_file.exists():
    long_description += "\n\n" + history_file.read_text(encoding="utf-8")

# All metadata is now in pyproject.toml
# This file is kept minimal for long_description combining
setup(long_description=long_description, long_description_content_type="text/x-rst")
