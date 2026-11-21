/**
 * @file model.h
 * This file provides the definition of ModelWeight and ModelStructure struct.
 * We define the detailed logic in C++ and then wrap it with C interface, such
 * that we can use the C interface in the PostgreSQL extension.
 * @note This header file is used by both C++ and C code. Therefore, we need to
 * wrap up the C++ code properly with #ifdef __cplusplus and extern "C" to
 * avoid the name mangling issue.
 */
#ifndef MODEL_H
#define MODEL_H

#include "neurstore/utils/tensor.h"


#ifdef __cplusplus
/*********************************** C++ ***********************************/
#include <vector>
#include <memory>
#include <optional>
#include <onnx.pb.h>

/**
 * Structure that warps the onnx::ModelProto.
 */
struct Model {
    onnx::ModelProto model;

    explicit Model();

    explicit Model(const onnx::ModelProto &model);

    explicit Model(onnx::ModelProto&& model);

    explicit Model(const char *path);

    explicit Model(const std::vector<uint8_t> &serialized_data);

    std::vector<uint8_t> serialize() const;
};


/**
 * Structure that holds the model weights.
 */
struct ModelWeight {
    std::vector<std::shared_ptr<TensorF64> > tensors;
    std::vector<std::string> names;
    std::vector<std::optional<IntTensorPacket::IntType> > int_types;

    ModelWeight(
        const std::vector<std::shared_ptr<TensorF64> > &tensors,
        const std::vector<std::string> &names,
        const std::vector<std::optional<IntTensorPacket::IntType> > &int_types
    );

    std::shared_ptr<TensorF64> getTensorF64(uint index) const;

    std::string getName(uint index) const;

    /**
     * Set the tensors and names
     * @param names names of the tensors
     * @param tensors tensors
     * @param int_types integer types of the tensors, if any
     */
    void set(
        const std::vector<std::string> &names,
        const std::vector<std::shared_ptr<TensorF64> > &tensors,
        const std::vector<std::optional<IntTensorPacket::IntType> > &int_types
    );

    /**
     * Get the number of tensors stored in ModelWeight
     * @return number of tensors
     */
    uint nTensors() const;

    bool isIntType(uint index) const;

    IntTensorPacket::IntType getIntType(uint index) const;
};

struct ModelWeightF16 {
    std::vector<std::shared_ptr<TensorF16> > tensor_f16;
    std::vector<std::string> names;
    std::vector<std::optional<IntTensorPacket::IntType> > int_types;

    ModelWeightF16(
        const std::vector<std::shared_ptr<TensorF16> > &tensor_f16,
        const std::vector<std::string> &names,
        const std::vector<std::optional<IntTensorPacket::IntType> > &int_types
    );

    std::shared_ptr<TensorF16> getTensorF16(uint index) const;

    std::string getName(uint index) const;

    /**
     * Set the tensors and names
     * @param names names of the tensors
     * @param tensor_f16 tensors
     * @param int_types integer types of the tensors, if any
     */
    void set(
        const std::vector<std::string> &names,
        const std::vector<std::shared_ptr<TensorF16> > &tensor_f16,
        const std::vector<std::optional<IntTensorPacket::IntType> > &int_types
    );

    /**
     * Get the number of tensors stored in ModelWeight
     * @return number of tensors
     */
    uint nTensors() const;

    bool isIntType(uint index) const;

    IntTensorPacket::IntType getIntType(uint index) const;
};


/**
 * Structure that holds the compressed model weights.
 */
struct TensorPage {
    std::vector<std::shared_ptr<TensorPacket> > tensor_packets;
    std::vector<std::string> names;

    TensorPage(
        const std::vector<std::shared_ptr<TensorPacket> > &tensor_packets,
        const std::vector<std::string> &names
    );

    TensorPage();

    std::shared_ptr<TensorPacket> getTensorPacket(uint index) const;

    std::string getName(uint index) const;

    /**
     * Set the tensor packets and ids
     * @param names names of the tensors
     * @param tensor_packets tensors
     */
    void set(
        const std::vector<std::string> &names,
        const std::vector<std::shared_ptr<TensorPacket> > &tensor_packets
    );

    /**
     * Get the number of tensors stored in TensorPage
     * @return number of tensors
     */
    uint nTensors() const;

    /**
     * Serialize the tensor page
     * the format is:
     * [packet_count: uint32_t][packet_1_size: uint32_t][packet_1: TensorPacket][packet_2_size: uint32_t]...[packet_n: TensorPacket]
     * [name_1_size: uint32_t][name_1: string][name_2_size: uint32_t][name_2: string]...[name_n: string]
     * @return serialized model weight compressed
     */
    std::vector<uint8_t> serialize() const;

    static TensorPage deserialize(const std::vector<uint8_t> &buffer);

    static TensorPage deserialize(const uint8_t *data, size_t size);
};


/**
 * Struct that holds the serialized structure of a model. (structure only)
 */
struct ModelStructure {
    std::vector<uint8_t> serialized_model_structure;
    std::vector<std::string> names; // tensor names

    explicit ModelStructure(
        const std::vector<uint8_t> &serialized_data,
        const std::vector<std::string> &names
    );

    explicit ModelStructure();

    void setTensorNames(const std::vector<std::string> &names);

    void setSerializedModelStructure(const std::vector<uint8_t> &serialized_data);

    void addTensorNames(const std::vector<std::string> &names);

    /**
     * Serialize the model structure
     * the serialized format is:
     * [serialized_model_structure_size: uint32_t][serialized_model_structure: vector<uint8_t>]
     * [names_count: uint32_t][name_1_size: uint32_t][name_1: string][name_2_size: uint32_t][name_2: string]...[name_n: string]
     * @return serialized model structure
     */
    std::vector<uint8_t> serialize() const;

    static ModelStructure deserialize(const std::vector<uint8_t> &data);

    static ModelStructure deserialize(const uint8_t *data, size_t size);
};

extern "C" {
#endif // __cplusplus
/************************************ C ************************************/

/**
 * struct that holds the Model defined in C++. m_ prefix is used to
 * represent all functions used for Model in C.
 */
typedef struct Model ModelC;

ModelC *m_create_empty_model();

ModelC *m_create_model_from_path(const char *path);

ModelC *m_create_model_from_serialized(const uint8_t *data, size_t n);

void m_destroy_model(ModelC *model);

char *m_serialize(ModelC *model, size_t *out_size);

/**
 * struct that holds the ModelWeight defined in C++. mw_ prefix is used to
 * represent all functions used for ModelWeight in C.
 */
typedef struct ModelWeight ModelWeightC;

ModelWeightC *mw_create_model_weight();

void mw_destroy_model_weight(ModelWeightC *model_weight);

uint mw_n_tensors(ModelWeightC *model_weight);

const char *mw_get_name(ModelWeightC *model_weight, uint index);

Tensor64C *mw_get_tensor(ModelWeightC *model_weight, uint index);

// void mw_set(ModelWeightC *model_weight, const char **names, Tensor64C **tensors, uint n);


/**
 * struct that holds TensorPage defined in C++. mwc_ prefix is used to
 * represent all functions used for TensorPage in C.
 */
typedef struct TensorPage TensorPageC;

TensorPageC *tp_create_tensor_page();

void tp_destroy_tensor_page(TensorPageC *tensor_page);

uint tp_n_tensors(TensorPageC *tensor_page);

const char *tp_get_name(TensorPageC *tensor_page, uint index);

TensorPacketC *tp_get_tensor_packet(TensorPageC *tensor_page, uint index);

void tp_set(
    TensorPageC *tensor_page,
    const char **names,
    TensorPacketC **tensor_packets,
    uint n
);

char *tp_serialize(TensorPageC *tensor_page, size_t *out_size);

TensorPageC *tp_deserialize(const uint8_t *data, size_t n);

/**
 * struct that holds the ModelStructure defined in C++. ms_ prefix is used to
 * represent all functions used for ModelStructure in C.
 */
typedef struct ModelStructure ModelStructureC;

ModelStructureC *ms_create_model_structure();

void ms_destroy_model_structure(ModelStructureC *model_structure);

void ms_set_tensor_names(ModelStructureC *model_structure, const char **names, uint n);

void ms_set_serialized_model_structure(ModelStructureC *model_structure, const uint8_t *data, uint n);

void ms_add_tensor_names(ModelStructureC *model_structure, const char **names, uint n);

char *ms_serialize(ModelStructureC *model_structure, size_t *out_size);

ModelStructureC *ms_deserialize(const uint8_t *data, size_t n);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // MODEL_H
