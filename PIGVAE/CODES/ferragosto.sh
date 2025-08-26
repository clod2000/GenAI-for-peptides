#!/bin/sh

# This script is used to run the main_fgvae.py with specific parameters.

# --- Configuration ---
# The base template for the config file
CONFIG_TEMPLATE="config.template.in"

# define the folder name
FOLDER="official_EGNN"


# 1. Create a descriptive name for this specific run
# This is CRITICAL for organizing your output files and logs
SIM_NAME="test"
#SIM_NAME="${SIM_NAME}_lr_${lr}_layers_${layers}_kl_min_${kl_min}"
echo "--- Starting run: ${SIM_NAME} ---"

# 2. Create a temporary config file for this run

TEMP_LOG="logs/${FOLDER}/${SIM_NAME}.log"
mkdir -p "$(dirname "$TEMP_LOG")"

python fmain_analysis.py --config "${CONFIG_TEMPLATE}" > "${TEMP_LOG}" 2>&1

# The '2>&1' redirects both standard output and standard error to the log file.

echo "--- Finished run: ${SIM_NAME}. Log saved to ${TEMP_LOG} ---"
echo ""


echo "All simulations finished"
