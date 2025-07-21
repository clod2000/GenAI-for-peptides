#!/bin/sh

# This script is used to run the main_fgvae.py with specific parameters.

# --- Configuration ---
# The base template for the config file
CONFIG_TEMPLATE="config.template.in"

# define the folder name
FOLDER="no_alignment"


# 1. Create a descriptive name for this specific run
# This is CRITICAL for organizing your output files and logs
SIM_NAME="sim_no_alignment"
#SIM_NAME="${SIM_NAME}_lr_${lr}_layers_${layers}_kl_min_${kl_min}"
echo "--- Starting run: ${SIM_NAME} ---"

# 2. Create a temporary config file for this run
TEMP_CONFIG="configs/${FOLDER}/${SIM_NAME}.in"
TEMP_LOG="logs/${FOLDER}/${SIM_NAME}.log"
mkdir -p "$(dirname "$TEMP_CONFIG")"
mkdir -p "$(dirname "$TEMP_LOG")"

# # 3. Use `sed` to modify the template and create the new config
# #    -e allows for multiple replacement commands
# sed -e "s/^LEARNING_RATE = .*/LEARNING_RATE: ${lr}/" \
#     -e "s/^NUM_ENC_LAYERS = .*/NUM_ENC_LAYERS: ${layers}/" \
#     -e "s/^NUM_DEC_LAYERS = .*/NUM_DEC_LAYERS: ${layers}/" \
#     -e "s/^NAME_SIMULATION = .*/NAME_SIMULATION: ${SIM_NAME}/" \
#     -e "s/^EPOCHS = .*/EPOCHS: 0/" \
#     -e "s/^MIN_KL = .*/MIN_KL: ${kl_min}/" \
#     -e "s/^LATENT_DIM = .*/LATENT_DIM: ${latent_dim}/" \
#     -e "s/^NAME_FOLDER = .*/NAME_FOLDER: ${FOLDER}/" \
#     -e "s/^ENCODER_POS_PROJECTION_DIM = .*/ENCODER_POS_PROJECTION_DIM: ${pos_proj}/" \
#     -e "s/wait_epochs = .*/wait_epochs: 15/" \
#     -e "s/annealing_epochs = .*/annealing_epochs: 150/" \
#     -e "s/beta_min = .*/beta_min: 0.000001/" \
#     -e "s/beta_max = .*/beta_max: 0.001/" \
sed -e "s/^LEARNING_RATE = .*/LEARNING_RATE: 0.00001/" \
    "${CONFIG_TEMPLATE}" > "${TEMP_CONFIG}"

# 4. Run your python script, passing the temporary config file
#    Redirect output to a log file for this specific run
python fmain.py --config "${TEMP_CONFIG}" > "${TEMP_LOG}" 2>&1

# The '2>&1' redirects both standard output and standard error to the log file.

echo "--- Finished run: ${SIM_NAME}. Log saved to ${TEMP_LOG} ---"
echo ""


echo "All simulations finished"
