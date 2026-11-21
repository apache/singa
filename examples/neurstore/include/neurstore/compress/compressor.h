#ifndef COMPRESSOR_H
#define COMPRESSOR_H

#include "neurstore/cache/index_cache_manager.h"
#include "neurstore/utils/global.h"


#ifdef __cplusplus
/*********************************** C++ ***********************************/

#include "neurstore/utils/model.h"
#include "neurstore/compress/builder/tensor_similarity_index.h"
#include "neurstore/compress/method/delta_quant_compress.h"
#include "neurstore/compress/strategy/strategy.h"


/**
 * PendingInsertion is a helper struct that holds tensors to be inserted into the index after compression.
 * It is used to support parallel compression of tensors.
 */
struct PendingInsertion {
    int dimension;
    int index_id;
    TensorType::VectorUInt8 tensor;
    double scale;
    double zero_point;
    size_t tensor_sequence_num; // sequence of the tensor in tensor_packets

    // allow move semantics for PendingInsertion
    PendingInsertion(PendingInsertion&&) noexcept = default;
    PendingInsertion& operator=(PendingInsertion&&) noexcept = default;
    // disable copy semantics for PendingInsertion
    PendingInsertion(const PendingInsertion&) = delete;
    PendingInsertion& operator=(const PendingInsertion&) = delete;

    PendingInsertion(
        int dimension,
        int index_id,
        TensorType::VectorUInt8 tensor,
        double scale,
        double zero_point,
        const size_t tensor_sequence_num
    ) : dimension(dimension),
        index_id(index_id),
        tensor(std::move(tensor)),
        scale(scale),
        zero_point(zero_point),
        tensor_sequence_num(tensor_sequence_num) {
    }
};

/**
 * Compressor class for compressing and decompressing models.
 * Handles the conversion between Model, ModelWeight, and ModelStructure.
 */
class Compressor {
    /**
     * Constructor for Compressor
     * @param tolerance Compression tolerance for quantization
     * @param dynamic Enable dynamic bit-width quantization
     * @param default_quantized_bit_width Default bit width for quantization, compulsory if dynamic is false
     * @param index_cache_manager Index cache manager for managing tensor similarity indices
     */
public:
    Compressor(
        double tolerance,
        bool dynamic,
        int default_quantized_bit_width,
        std::shared_ptr<IndexCacheManager> index_cache_manager
    );

    /**
     * Compress a model
     * @param model the model to compress
     * @return ModelStructure and ModelWeight
     */
    std::pair<ModelStructure, TensorPage> compress(
        const Model &model
    );

    /**
     * Decompress a model
     * @param model_structure model structure
     * @param tensor_page tensor page
     * @return
     */
    Model decompress(
        const ModelStructure &model_structure,
        const TensorPage &tensor_page
    );

    Model decompressFloat16(
        const ModelStructure &model_structure,
        const TensorPage &tensor_page
    );

    /**
     * Compress a single tensor
     * @param tensor the tensor to compress
     * @param tensor_index the index of the tensor in the tensor_packets
     * @param pending_insertion_tensors tensors that will be inserted into the index after compression
     * @return The compressed tensor in TensorPacket
     */
    std::shared_ptr<TensorPacket> compressTensor(
        const std::shared_ptr<TensorF64> &tensor,
        uint tensor_index,
        std::unordered_map<int, std::vector<PendingInsertion>
        > &pending_insertion_tensors
    );

    /**
     * Decompress a single tensor
     * @param tensor_packet the compressed tensor packet
     * @param index the tensor similarity index
     * @return The decompressed tensor
     */
    std::shared_ptr<TensorF64>
    decompressTensor(
        const std::shared_ptr<TensorPacket> &tensor_packet,
        const std::shared_ptr<VectorUInt8Index> &index
    );

    std::shared_ptr<Model> reconstructModelUInt8(
        const TensorPage &model_weight,
        const ModelStructure &model_structure
    ) const;

    std::shared_ptr<Model> reconstructModelUInt8WithDelta(
        const TensorPage &model_weight,
        const ModelStructure &model_structure
    ) const;

private:
    double tolerance_{};
    bool dynamic_{};
    int default_quantized_bit_width_{};
    DeltaQuantCompress delta_quant_compress_;
    std::shared_ptr<SelectionStrategy> selection_strategy_;
    std::shared_ptr<IndexCacheManager> index_cache_manager_;

    /**
     * Helper function to extract tensors from a model.
     * @param model The ONNX model
     * @return A ModelWeight containing all tensors and their names
     */
    static std::shared_ptr<ModelWeight> extractModelTensors(const Model &model);

    /**
     * Reconstruct a model from ModelWeight and ModelStructure
     * @param model_weight the decompressed model weight
     * @param model_structure the model structure
     * @return The reconstructed ONNX model
     */
    static std::shared_ptr<Model> reconstructModel(
        const ModelWeight &model_weight,
        const ModelStructure &model_structure
    );

    static std::shared_ptr<Model> reconstructModelF16(
        const ModelWeightF16 &model_weight,
        const ModelStructure &model_structure
    );
};

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        return std::hash<T1>()(p.first) ^ (std::hash<T2>()(p.second) << 1);
    }
};

extern "C" {
#endif // __cplusplus
/************************************ C ************************************/

// typedef struct Compressor CompressorC;
//
// CompressorC *cp_create_compressor(
//     double tolerance,
//     int dynamic,
//     size_t index_cache_size,
//     int default_quantized_bit_width,
//     const char *store_path
// );
//
// void cp_destroy_compressor(CompressorC *compressor);
//
// void cp_compress(
//     CompressorC *compressor,
//     ModelC *model,
//     ModelStructureC *model_structure_out,
//     TensorPageC *tensor_page_out
// );
//
// void cp_decompress(
//     CompressorC *compressor,
//     ModelStructureC *model_structure,
//     TensorPageC *tensor_page,
//     ModelC *model_out
// );
//
// void cp_evict_cache(CompressorC *compressor);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif //COMPRESSOR_H
