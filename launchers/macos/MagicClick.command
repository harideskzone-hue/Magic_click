#!/bin/bash
# MagicClick.command
# Double-clickable macOS launcher script.

# Ensure we run from the project root
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR/../.."

# Run the bootstrap (which creates venv and shows UI)
python3 bootstrap.py
