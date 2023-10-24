

CREATE OR REPLACE
PROCEDURE model_selection_workloads(
    dataset TEXT,               --dataset name
    selected_columns TEXT[],    --used columns
    N INTEGER,                  --explore N models
    K INTEGER,                  --keep K models
    config_file TEXT            --config file path
)
LANGUAGE plpgsql
AS $$
DECLARE
    -- global inputs/outputs
    result_status TEXT;
    column_list TEXT;

BEGIN
    -- combine the columns into a string
    column_list := array_to_string(selected_columns, ', ');
    EXECUTE format('
                WITH batch_rows AS (
                    SELECT %s
                    FROM %I
                    ORDER BY RANDOM()
                )
                SELECT model_selection_workloads(
                    json_agg(row_to_json(t))::text, %s, %s, %L
                )
                FROM batch_rows AS t', column_list, dataset, N, K, config_file) INTO result_status;
    RAISE NOTICE '1. model_selection result: %', result_status;
END; $$;
