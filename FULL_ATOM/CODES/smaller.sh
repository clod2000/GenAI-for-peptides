#!/bin/sh

# This script is used to run the main_fgvae.py with specific parameters.
# version use to run test with smaller models

# --- Configuration ---
# The base template for the config file
CONFIG_TEMPLATE="config.template.in"

# define the folder name
FOLDER="small_models_1"

for layers in 3 4 5; do
    for latent_dim in 1 2 3 4; do
        

        # 1. Create a descriptive name for this specific run
        # This is CRITICAL for organizing your output files and logs
        SIM_NAME="sim_latent_dim_${latent_dim}_layers_${layers}"
        #SIM_NAME="${SIM_NAME}_lr_${lr}_layers_${layers}_kl_min_${kl_min}"
        echo "--- Starting run: ${SIM_NAME} ---"

        # 2. Create a temporary config file for this run
        TEMP_CONFIG="configs/${FOLDER}/${SIM_NAME}.in"
        TEMP_LOG="logs/${FOLDER}/${SIM_NAME}.log"
        mkdir -p "$(dirname "$TEMP_CONFIG")"
        mkdir -p "$(dirname "$TEMP_LOG")"

        # # 3. Use `sed` to modify the template and create the new config
        # #    -e allows for multiple replacement commands
        sed -e "s/^NUM_ENC_LAYERS = .*/NUM_ENC_LAYERS: ${layers}/" \
            -e "s/^NUM_DEC_LAYERS = .*/NUM_DEC_LAYERS: ${layers}/" \
            -e "s/^NAME_SIMULATION = .*/NAME_SIMULATION: ${SIM_NAME}/" \
            -e "s/^LATENT_DIM = .*/LATENT_DIM: ${latent_dim}/" \
            -e "s/^NAME_FOLDER = .*/NAME_FOLDER: ${FOLDER}/" \
            -e "s/^ENCODER_POS_PROJECTION_DIM = .*/ENCODER_POS_PROJECTION_DIM: ${latent_dim}/" \
            "${CONFIG_TEMPLATE}" > "${TEMP_CONFIG}"

        # 4. Run your python script, passing the temporary config file
        #    Redirect output to a log file for this specific run
        python fmain.py --config "${TEMP_CONFIG}" > "${TEMP_LOG}" 2>&1

        # The '2>&1' redirects both standard output and standard error to the log file.

        echo "--- Finished run: ${SIM_NAME}. Log saved to ${TEMP_LOG} ---"
        echo ""
    done
done

echo "All simulations finished"
