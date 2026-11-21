#ifndef ROUTER_H
#define ROUTER_H

#include "neurstore/utils/tensor.h"


template<typename T>
class Router {
public:
    virtual ~Router() = default;

    explicit Router(int dimension, int nindex = 0, int bit_width = 32)
        : dimension_(dimension), nindex_(nindex), bit_width_(bit_width) {
        if (dimension <= 0) {
            throw std::invalid_argument("Router::Router: dimension must be positive");
        }
    }

    virtual int route(const T &vector) = 0;

    virtual void add(const T &vector) = 0;

    virtual void update(int index_id, const T &vector, int num_of_vectors) = 0;

    virtual void serialize(std::ostream &out) const = 0;

    int getDimension() const {
        return dimension_;
    }

    void setDimension(int dimension) {
        if (dimension <= 0) {
            throw std::invalid_argument("Router::setDimension: dimension must be positive");
        }
        dimension_ = dimension;
    }

    int getNumOfIndex() const {
        return nindex_;
    }

    void setNumOfIndex(int nindex) {
        if (nindex < 0) {
            throw std::invalid_argument("Router::setNumOfIndex: nindex must be non-negative");
        }
        nindex_ = nindex;
    }

    void setBitWidth(int bit_width) {
        if (bit_width != 4 && bit_width != 8 && bit_width != 16 && bit_width != 32 && bit_width != 64) {
            throw std::invalid_argument("Router::setBitWidth: unsupported bit width");
        }
        bit_width_ = bit_width;
    }

    int getBitWidth() const {
        return bit_width_;
    }

    int getNextIndexId() const {
        return nindex_;
    }

    int getCurrentIndexId() const {
        return nindex_-1;
    }

    int incrementNumOfIndex() {
        return ++nindex_;
    }

private:
    int dimension_;
    int nindex_;
    int bit_width_;
};

typedef Router<TensorType::VectorFloat64> VectorFloat64Router;

#endif //ROUTER_H
