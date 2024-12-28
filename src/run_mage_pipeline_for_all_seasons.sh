#!/bin/bash

# Set the project path and pipeline UUID
PROJECT_PATH="/Users/cb/src/nba_mvp_ml"
PIPELINE_UUID="nba_data_scraping"

# Loop through each year from 1980 to 2023
# for year in $(seq 1980 2023); do
for year in $(seq 1993 2023); do
  echo "Running pipeline for season: $year"
  
  # Run the Mage pipeline
  mage run "$PROJECT_PATH" "$PIPELINE_UUID" --runtime-vars "{\"season\": $year}"
  
  # Check if the last command was successful
  if [ $? -ne 0 ]; then
    echo "Pipeline failed for season: $year. Exiting loop."
    exit 1
  fi
  
  echo "Pipeline completed for season: $year"
done

echo "All years processed successfully!"