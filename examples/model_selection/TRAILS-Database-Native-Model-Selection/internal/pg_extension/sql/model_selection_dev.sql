

CREATE OR REPLACE
PROCEDURE model_selection_sp(
    dataset TEXT,               --dataset name
    selected_columns TEXT[],    --used columns
    budget TEXT,                --user given time budget
    batch_size INTEGER,         --batch size, for profiling, filtering
    config_file TEXT            --config file path
)
LANGUAGE plpgsql
AS $$
DECLARE
    -- global inputs/outputs
    result_status TEXT;
    column_list TEXT;

    -- UDF outputs
    score_time TEXT;
    train_time TEXT;
    coordinator_k integer;
    coordinator_u integer;
    coordinator_n integer;
BEGIN
    -- combine the columns into a string
    column_list := array_to_string(selected_columns, ', ');

    -- 1. Profiling time to score a model with TFMEM
    EXECUTE format('
                WITH batch_rows AS (
                    SELECT %s
                    FROM %I
                    ORDER BY RANDOM()
                    LIMIT %s OFFSET 0
                )
                SELECT profiling_filtering_phase(
                    json_agg(row_to_json(t))::text, %L
                )
                FROM batch_rows AS t', column_list, dataset, batch_size, config_file) INTO result_status;
    score_time := json_extract_path_text(result_status::json, 'time');
    RAISE NOTICE '1. profiling_filtering_phase, get score_time: %', score_time;

    -- 2. Profiling time of training a model for one epoch
    EXECUTE format('
                WITH batch_rows AS (
                    SELECT %s
                    FROM %I
                    ORDER BY RANDOM()
                    LIMIT %s OFFSET 0
                )
                SELECT profiling_refinement_phase(
                    json_agg(row_to_json(t))::text, %L
                )
                FROM batch_rows AS t', column_list, dataset, batch_size, config_file) INTO result_status;
    train_time := json_extract_path_text(result_status::json, 'time');
    RAISE NOTICE '2. profiling_refinement_phase, get train_time: %', train_time;

    -- 3. Coordinator to get N, K ,U
    EXECUTE format('SELECT "coordinator"(%L, %L, %L, false, %L)', score_time, train_time, budget, config_file) INTO result_status;

    coordinator_k := (json_extract_path_text(result_status::json, 'k'))::integer;
    coordinator_u := (json_extract_path_text(result_status::json, 'u'))::integer;
    coordinator_n := (json_extract_path_text(result_status::json, 'n'))::integer;
    RAISE NOTICE '3. coordinator result: k = %, u = %, n = %', coordinator_k, coordinator_u, coordinator_n;

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
            FROM batch_rows AS t', column_list, dataset, batch_size, coordinator_n, coordinator_k, config_file) INTO result_status;
    RAISE NOTICE '4. run filtering phase, k models = %', result_status;

END; $$;
