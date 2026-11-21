#ifndef STRATEGY_H
#define STRATEGY_H

#include <vector>
#include <memory>

#include "neurstore/utils/tensor.h"


/**
 * Interface for a representative tensor selection strategy
 */
class SelectionStrategy {
public:
    SelectionStrategy(const double tolerance, const int nbits, const std::vector<int64_t> &shape)
        : tolerance_(tolerance), nbits_(nbits), shape_(shape) {
    }

    virtual ~SelectionStrategy() = default;

    /**
     * Get the probability of selecting a tensor as a representative tensor
     * @param representative_tensor The representative tensor
     * @param tensor The tensor to select
     * @return The probability of selecting the tensor
     */
    virtual double getProbability(
        const TensorType::VectorFloat64 &representative_tensor,
        const TensorType::VectorFloat64 &tensor
    ) = 0;

    /**
     * Select a tensor as a representative tensor
     * @param representative_tensor The representative tensor
     * @param tensor The tensor to select
     * @return True if the tensor is selected, false otherwise
     */
    bool selectAsRepresentative(
        const TensorType::VectorFloat64 &representative_tensor,
        const TensorType::VectorFloat64 &tensor
    );

    /**
     *Serialize the strategy into a byte array, to be implemented by subclasses
     * @return Byte array
     */
    virtual std::vector<uint8_t> serialize() const = 0;

    /**
     * @brief Serialize the strategy into a byte array
     * @param strategy
     * @return Serialized byte array
     */
    static std::vector<uint8_t> serialize(const std::shared_ptr<SelectionStrategy> &strategy);

    /**
     * @brief Load the strategy from a byte array
     * @param data The byte array
     * @return SelectionStrategy
     */
    static std::shared_ptr<SelectionStrategy> deserialize(const std::vector<uint8_t> &data);

protected:
    double tolerance_;
    int nbits_;
    std::vector<int64_t> shape_;
};

#endif //STRATEGY_H
