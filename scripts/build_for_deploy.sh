#!/usr/bin/env bash
# ==============================================================================
# SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization
# Build script for Render deployment / headless cloud container startup.
# Generates all synthetic data, trained models, optimization sweeps, and reports
# required by the executive dashboard from a clean git checkout.
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo "[SIH26138 BUILD] Starting automated artifact generation pipeline..."
echo "======================================================================"

# Ensure directories exist
mkdir -p data/raw data/processed artifacts/models artifacts/metrics outputs/reports outputs/logs outputs/figures

# Run complete end-to-end maritime intelligence & green fleet optimization pipeline
python scripts/run_full_pipeline.py --rows 10000


echo "======================================================================"
echo "[SIH26138 BUILD] Deployment build completed successfully!"
echo "======================================================================"
