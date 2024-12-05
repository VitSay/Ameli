#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <string>

// Enum for computation device
enum class Device {
    CPU,
    GPU // Placeholder, GPU implementation not detailed
};

template <typename T>
class Tensor {
public:
    // Constructors
    explicit Tensor(const std::vector<size_t>& shape, T value = T(), Device device = Device::CPU)
        : shape_(shape), data_(calcSize(shape), value), device_(device) {}

    Tensor() = default;

    // Access element at specific indices
    T& operator()(const std::vector<size_t>& indices) {
        size_t index = getFlatIndex(indices);
        return data_[index];
    }

    const T& operator()(const std::vector<size_t>& indices) const {
        size_t index = getFlatIndex(indices);
        return data_[index];
    }

    // Basic arithmetic operations
    Tensor<T> operator+(const Tensor<T>& other) const {
        validateShapeBroadcast(other);
        Tensor<T> result(broadcastedShape(other), T(), device_);
        for (size_t i = 0; i < result.size(); ++i) {
            result.data_[i] = this->dataAtBroadcasted(i, *this) + this->dataAtBroadcasted(i, other);
        }
        return result;
    }

    Tensor<T> operator-(const Tensor<T>& other) const {
        validateShapeBroadcast(other);
        Tensor<T> result(broadcastedShape(other), T(), device_);
        for (size_t i = 0; i < result.size(); ++i) {
            result.data_[i] = this->dataAtBroadcasted(i, *this) - this->dataAtBroadcasted(i, other);
        }
        return result;
    }

    Tensor<T> operator*(const T& scalar) const {
        Tensor<T> result(shape_, T(), device_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] * scalar;
        }
        return result;
    }

    Tensor<T> operator/(const T& scalar) const {
        if (scalar == T(0)) {
            throw std::invalid_argument("Division by zero.");
        }
        Tensor<T> result(shape_, T(), device_);
        for (size_t i = 0; i < data_.size(); ++i) {
            result.data_[i] = data_[i] / scalar;
        }
        return result;
    }

    // Dot product (only for 1D tensors)
    T dot(const Tensor<T>& other) const {
        if (shape_.size() != 1 || other.shape_.size() != 1 || shape_[0] != other.shape_[0]) {
            throw std::invalid_argument("Dot product requires 1D tensors of the same size.");
        }
        T result = T();
        for (size_t i = 0; i < data_.size(); ++i) {
            result += data_[i] * other.data_[i];
        }
        return result;
    }

    // Reshape tensor
    void reshape(const std::vector<size_t>& newShape) {
        if (calcSize(newShape) != data_.size()) {
            throw std::invalid_argument("New shape must match the total size.");
        }
        shape_ = newShape;
    }

    // Getters
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return data_.size(); }
    Device device() const { return device_; }

    // Print tensor (for debugging)
    void print() const {
        std::cout << "Tensor (device: " << (device_ == Device::CPU ? "CPU" : "GPU") << ", shape: [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            std::cout << shape_[i];
            if (i != shape_.size() - 1) std::cout << ", ";
        }
        std::cout << "]): [";
        for (size_t i = 0; i < data_.size(); ++i) {
            std::cout << data_[i];
            if (i != data_.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }

private:
    std::vector<size_t> shape_;
    std::vector<T> data_;
    Device device_;

    size_t calcSize(const std::vector<size_t>& shape) const {
        return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<>());
    }

    size_t getFlatIndex(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Invalid number of indices.");
        }
        size_t flatIndex = 0;
        size_t stride = 1;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds.");
            }
            flatIndex += indices[i] * stride;
            stride *= shape_[i];
        }
        return flatIndex;
    }

    void validateShapeBroadcast(const Tensor<T>& other) const {
        auto maxDims = std::max(shape_.size(), other.shape_.size());
        for (size_t i = 0; i < maxDims; ++i) {
            size_t dimA = i < shape_.size() ? shape_[shape_.size() - 1 - i] : 1;
            size_t dimB = i < other.shape_.size() ? other.shape_[other.shape_.size() - 1 - i] : 1;
            if (dimA != dimB && dimA != 1 && dimB != 1) {
                throw std::invalid_argument("Shapes cannot be broadcasted.");
            }
        }
    }

    std::vector<size_t> broadcastedShape(const Tensor<T>& other) const {
        std::vector<size_t> result;
        auto maxDims = std::max(shape_.size(), other.shape_.size());
        for (size_t i = 0; i < maxDims; ++i) {
            size_t dimA = i < shape_.size() ? shape_[shape_.size() - 1 - i] : 1;
            size_t dimB = i < other.shape_.size() ? other.shape_[other.shape_.size() - 1 - i] : 1;
            result.push_back(std::max(dimA, dimB));
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    T dataAtBroadcasted(size_t index, const Tensor<T>& source) const {
        std::vector<size_t> indices;
        size_t stride = 1;
        for (int i = source.shape_.size() - 1; i >= 0; --i) {
            indices.push_back((index / stride) % source.shape_[i]);
            stride *= source.shape_[i];
        }
        std::reverse(indices.begin(), indices.end());
        for (size_t i = 0; i < indices.size(); ++i) {
            if (shape_[i] != source.shape_[i] && source.shape_[i] == 1) {
                indices[i] = 0;
            }
        }
        return source(indices);
    }
};

#endif // TENSOR_HPP