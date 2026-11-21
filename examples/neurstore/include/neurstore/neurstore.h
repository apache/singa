#ifndef NEURSTORE_H
#define NEURSTORE_H

#ifdef __cplusplus
/*********************************** C++ ***********************************/
#include <memory>
#include <onnx.pb.h>
#include <string>
#include <vector>
#include <any>

#include "compress/compressor.h"
#include "inference/inference_utils.h"

namespace fs = std::filesystem;


class NeurStore {
public:
    explicit NeurStore(std::string &storePath, const std::shared_ptr<IndexCacheManager> &indexCacheManager);

    bool saveModel(
        const std::string &modelName,
        double tolerance,
        const std::string &modelPath
    ) const;

    bool saveModels(
        const std::vector<std::string> &modelNames,
        double tolerance,
        const std::string &folderPath
    ) const;

    bool saveModelsFromPaths(
        const std::vector<std::string> &modelPaths,
        double tolerance
    ) const;

    std::pair<double, double> evaluateTolerance(
        double tolerance,
        const std::string &modelPath,
        int load_mode,
        const InferenceOptions &options
    ) const;

    std::shared_ptr<onnx::ModelProto> loadModel(const std::string &modelName) const;

    std::shared_ptr<onnx::ModelProto> loadModelAsFloat16(
        const std::string &modelName
    ) const;

    std::shared_ptr<onnx::ModelProto> loadModelAsInt8(
        const std::string &modelName
    ) const;

    std::shared_ptr<onnx::ModelProto> loadModelAsInt8WithDelta(
        const std::string &modelName
    ) const;

    void inference(const InferenceOptions& options, std::vector<std::any>& out) const;
    void cleanCache() const;

    uintmax_t totalSize() const;
    uintmax_t modelSize() const;
    uintmax_t indexSize() const;
    uint32_t modelCount() const;
    uint32_t indexCount() const;
    std::string summary() const;

private:
    std::string storePath_;

    std::shared_ptr<IndexCacheManager> indexCacheManager_;

    std::pair<std::shared_ptr<onnx::ModelProto>, double> saveModelDryRun(
        double tolerance,
        const std::string &modelPath,
        int load_mode
    ) const;

    std::shared_ptr<onnx::ModelProto> loadModelAsFloat16(const std::string &modelName, Compressor &compressor) const;

    std::shared_ptr<onnx::ModelProto> loadModel(const std::string &modelName, Compressor &compressor) const;

    std::shared_ptr<onnx::ModelProto> loadModelAsInt8(const std::string &modelName, Compressor &compressor) const;

    std::shared_ptr<onnx::ModelProto> loadModelAsInt8WithDelta(const std::string &modelName, Compressor &compressor) const;

    void inference(
        const std::shared_ptr<onnx::ModelProto> &model,
        const InferenceOptions &options,
        std::vector<std::any> &out
    ) const;

    bool saveModelInternal(
        const std::string &modelName,
        std::vector<unsigned char> &serializedModelStructure,
        std::vector<unsigned char> &serializedTensorPage
    ) const;

    void loadModelMetaInternal(
        const std::string &modelName,
        ModelStructure &modelStructure,
        TensorPage &tensorPage
    ) const;
};

extern "C" {
#endif // __cplusplus
/*********************************** C ***********************************/
#include "neurstore/utils/model.h"
#include "neurstore/cache/index_cache_manager.h"
#include "neurstore/inference/inference_utils.h"

bool set_parallelism(int parallelism);

int get_parallelism();

typedef struct NeurStore NeurStoreC;

NeurStoreC *ns_create(const char *store_path, IndexCacheManagerC *cache_mgr);

void ns_destroy(NeurStoreC *nstore);

ModelC* ns_load_model_internal(NeurStoreC* ns, const char* model_name);

ModelC* ns_load_model_intermal_uint8(
    NeurStoreC* ns,
    const char* model_name
);

ModelC* ns_load_model_intermal_uint8_delta(
    NeurStoreC* ns,
    const char* model_name
);

ModelC* ns_load_model_intermal_float16(
    NeurStoreC* ns,
    const char* model_name
);

bool ns_save_model_internal(
    NeurStoreC *ns,
    const char *model_name,
    double tolerance,
    const char *model_path
);

bool ns_save_models_internal(
    NeurStoreC *ns,
    const char **model_names,
    int model_count,
    double tolerance,
    const char *folder_path
);

bool ns_save_model_dry_run_internal(
    NeurStoreC* ns,
    double tolerance,
    const char* model_path,
    int load_mode,
    InferenceOptionsC *options,
    double* ratio_out,
    double* delta_performance_out
);

int ns_inference_internal(
    NeurStoreC *ns,
    InferenceOptionsC *options,
    const char **err_msg_out
);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif //NEURSTORE_H
