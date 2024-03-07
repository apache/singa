#!/bin/bash

# Check for proper number of command line args
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <data_path> <db_name>"
    exit 1
fi

# Configurations
DATA_PATH="$1"
DB_NAME="$2"

# Connection details
HOST="localhost"
PORT="5432"
USERNAME="postgres"
DBNAME="model_slicing"

# Create the database
echo "Creating database..."
createdb -h $HOST -p $PORT -U $USERNAME $DBNAME



echo "Data load complete."
