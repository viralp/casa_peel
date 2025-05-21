#!/bin/bash

# Define CASA root directory (adjust if needed)
CASA_ROOT="$PWD/casa-6.6.4-34-py3.8.el8"

# Path to CASA Python
CASA_PYTHON="$CASA_ROOT/lib/py/bin/python3.8"

# Validate CASA Python exists
if [[ ! -x "$CASA_PYTHON" ]]; then
    echo "Error: CASA Python not found at $CASA_PYTHON"
    exit 1
fi

# Look for goquartical inside CASA environment
GOQUARTICAL_PATH=$(find "$CASA_ROOT" -type f -name "goquartical" -executable | head -n 1)

# Check if we found it
if [[ -z "$GOQUARTICAL_PATH" ]]; then
    echo "goquartical not found inside CASA installation."
    exit 1
fi

# Backup the original script
cp "$GOQUARTICAL_PATH" "${GOQUARTICAL_PATH}.bak"

# Replace the shebang line with CASA's Python
sed -i "1s|^.*$|#!$CASA_PYTHON|" "$GOQUARTICAL_PATH"

# Done
echo "goquartical inside CASA updated:"
echo "$GOQUARTICAL_PATH -> #!$CASA_PYTHON"

