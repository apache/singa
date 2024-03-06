#!/bin/bash

# Those cmds will triggered after docker run .

# Compile code, and run postgresql
cd /project/singa/examples/model_selection/TRAILS-Database-Native-Model-Selection/internal/pg_extension || exit
/bin/bash -c "source $HOME/.cargo/env && echo '\q' | cargo pgrx run --release"

# Wait for PostgreSQL to become available
until psql -h localhost -p 28814 -U postgres -d pg_extension -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

# Run setup commands
psql -h localhost -p 28814 -U postgres -d pg_extension -c "CREATE EXTENSION pg_extension;"
psql -h localhost -p 28814 -U postgres -d pg_extension -f /project/singa/examples/model_selection/TRAILS-Database-Native-Model-Selection/internal/pg_extension/sql/model_selection_cpu.sql
# Load example dataset into database
bash /project/singa/examples/model_selection/TRAILS-Database-Native-Model-Selection/internal/ml/model_selection/TRAILS-Database-Native-Model-Selection/scripts/database/load_data_to_db.sh /project/singa/examples/model_selection/TRAILS-Database-Native-Model-Selection/dataset/frappe frappe

# Continue with the rest of your container's CMD
tail -f /dev/null

echo "Done!"
