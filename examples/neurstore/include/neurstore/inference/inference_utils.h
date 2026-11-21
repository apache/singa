#ifndef INFERENCE_UTILS_H
#define INFERENCE_UTILS_H

#ifdef __cplusplus
/*********************************** C++ ***********************************/
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <tokenizers_cpp.h>
#include <fstream>
#include <iostream>


enum LoadMode {
    Float32 = 0,
    Int8,
    Int8Delta,
    Float16
};

enum class Task {
    SequenceClassification,
    ImageClassification,
    ObjectDetection,
    TabularClassification,
    TabularRegression,
};

struct InferenceOptions {
    std::string model_id;
    std::string query;
    std::string pg_dsn; // postgres connection string
    std::vector<std::string> input_columns;
    std::string output_column;
    int batch_size;
    std::string tokenizer_path;
    std::string pad_token;
    std::string eos_token;
    std::string bos_token;
    int max_input_len;
    int max_output_len;
    LoadMode load_mode;
    Task task;
    bool use_gpu;
};

enum class ColumnType {
    String,
    Int,
    Float
};

struct RawColumnData {
    ColumnType type;
    std::string name;
    std::vector<std::string> string_data;   // For string columns
    std::vector<int64_t> int64_data;        // For int64 columns
    std::vector<int32_t> int32_data;        // For int32 columns
    std::vector<float> float_data;          // For float columns
    std::vector<bool> boolean_data;    // For boolean columns
};

struct RawBatchData {
    std::vector<RawColumnData> columns;
};

struct ProcessedBatchData {
    virtual ~ProcessedBatchData() = default;
    virtual size_t batchSize() const = 0;
    virtual std::vector<std::pair<std::string, Ort::Value>> toOrtValues(Ort::MemoryInfo& memory_info) const = 0;
};

inline std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
        std::cerr << "LoadBytesFromFile: Fail to open " << path << std::endl;
        exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
}

extern "C" {
#endif // __cplusplus
/*********************************** C ***********************************/

typedef struct InferenceOptions InferenceOptionsC;

InferenceOptionsC *ioc_create(const char   *model_id,
           const char   *query,
           const char   *pg_dsn,
           const char  **input_columns, int n_input_columns,
           const char   *output_column,
           int           batch_size,
           const char   *tokenizer_path,
           const char   *pad_token,
           const char   *eos_token,
           const char   *bos_token,
           int           max_input_len,
           int           max_output_len,
           int           load_mode,
           int           task,
           int           use_gpu);

void ioc_destroy(InferenceOptionsC *options);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif //INFERENCE_UTILS_H
