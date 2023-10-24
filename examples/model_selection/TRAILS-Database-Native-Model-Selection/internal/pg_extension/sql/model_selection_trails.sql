

CREATE OR REPLACE
PROCEDURE model_selection_end2end(
    dataset TEXT,               --dataset name
    selected_columns TEXT[],    --used columns
    budget TEXT,                --user given time budget
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
                SELECT model_selection_trails(
                    json_agg(row_to_json(t))::text, %L, %L
                )
                FROM batch_rows AS t', column_list, dataset, budget, config_file) INTO result_status;
    RAISE NOTICE '1. model_selection result: %', result_status;
END; $$;
