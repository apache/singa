

CREATE OR REPLACE
PROCEDURE model_selection_sp(
    dataset TEXT,               --dataset name
    selected_columns TEXT[],    --used columns
    N INTEGER,                  --number of models to evaluate
    batch_size INTEGER,         --batch size, for profiling, filtering
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

    -- 4. Run filtering phase to get top K models.
    EXECUTE format('
                WITH batch_rows AS (
                    SELECT %s
                    FROM %I
                    ORDER BY RANDOM()
                    LIMIT %s OFFSET 0
                )
                SELECT filtering_phase(
                    json_agg(row_to_json(t))::text, %s, %s, %L
                )
                FROM batch_rows AS t', column_list, dataset, batch_size, N, 1, config_file) INTO result_status;
    RAISE NOTICE '4. run filtering phase, k models = %', result_status;

END; $$;
