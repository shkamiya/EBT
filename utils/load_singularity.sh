#!/bin/bash

# Check if 'apptainer' or 'singularity' is available
if command -v apptainer >/dev/null 2>&1; then
    echo "Apptainer is available."
elif command -v singularity >/dev/null 2>&1; then
    echo "Singularity is available."
else
    echo "Neither Apptainer nor Singularity is available. Attempting to load Apptainer."

    # Try to load 'apptainer'
    module load apptainer >/dev/null 2>&1

    # Check if loading 'apptainer' was successful
    if command -v apptainer >/dev/null 2>&1; then
        echo "Successfully loaded Apptainer."
    else
        echo "Failed to load Apptainer. Attempting to load Singularity."
        
        # Try to load 'singularity'
        module load singularity >/dev/null 2>&1

        # Check if loading 'singularity' was successful
        if command -v singularity >/dev/null 2>&1; then
            echo "Successfully loaded Singularity."
        else
            echo "Failed to load both Apptainer and Singularity."
        fi
    fi
fi
