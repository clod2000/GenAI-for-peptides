#!/bin/sh

# This script is used to run the main_fgvae.py with specific parameters.

# --- Configuration ---
# The base template for the config file
CONFIG_TEMPLATE="config.template.in"

# --- Experiment Loop ---
# Let's iterate over a few learning rates
for lr in  0.0001 0.00001; do
  # And for each learning rate, let's test a few layer counts
  for layers in 9; do

    # 1. Create a descriptive name for this specific run
    # This is CRITICAL for organizing your output files and logs
    SIM_NAME="sim_lr_${lr}_layers_${layers}"
    echo "--- Starting run: ${SIM_NAME} ---"

    # 2. Create a temporary config file for this run
    TEMP_CONFIG="configs/config_run_${SIM_NAME}.in"
    # Create the directory for configs if it doesn't exist
    mkdir -p "$(dirname "$TEMP_CONFIG")"
    # Create the logs directory if it doesn't exist.
    mkdir -p logs/low_KL/
    # 3. Use `sed` to modify the template and create the new config
    #    -e allows for multiple replacement commands
    sed -e "s/^LEARNING_RATE = .*/LEARNING_RATE: ${lr}/" \
        -e "s/^NUM_ENC_LAYERS = .*/NUM_ENC_LAYERS: ${layers}/" \
        -e "s/^NUM_DEC_LAYERS = .*/NUM_DEC_LAYERS: ${layers}/" \
        -e "s/^NAME_SIMULATION = .*/NAME_SIMULATION: ${SIM_NAME}/" \
        -e "s/^EPOCHS = .*/EPOCHS: 50/" \
        "${CONFIG_TEMPLATE}" > "${TEMP_CONFIG}"

    # 4. Run your python script, passing the temporary config file
    #    Redirect output to a log file for this specific run
    python fmain.py --config "${TEMP_CONFIG}" > "logs/low_KL/${SIM_NAME}.log" 2>&1
    
    # The '2>&1' redirects both standard output and standard error to the log file.
    

    echo "--- Finished run: ${SIM_NAME}. Log saved to logs/${SIM_NAME}.log ---"
    echo ""

  done
done

echo "All simulations finished, cleaning config dir."
# Clean up the temporary config files
rm -f configs/config_run_*.in
