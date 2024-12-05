#include <iostream>
#include "tensor.hpp"

int main() {
    // Создание двух тензоров 1D (вектор)
    Tensor<float> vec1({3}, 1.0f); // Вектор длины 3, заполненный 1.0
    Tensor<float> vec2({3}, 2.0f); // Вектор длины 3, заполненный 2.0

    // Скалярное произведение
    float dotProduct = vec1.dot(vec2);
    std::cout << "Dot product: " << dotProduct << std::endl; // Ожидаемый результат: 6.0 (1*2 + 1*2 + 1*2)

    // Создание 2D тензора и заполнение значениями
    Tensor<int> mat1({2, 3}, 5); // Матрица 2x3, заполненная 5
    mat1({0, 0}) = 10;          // Изменяем значение в [0, 0]

    // Вывод матрицы
    mat1.print();

    // Broadcasting: сложение матрицы и вектора
    Tensor<int> mat2 = mat1 + Tensor<int>({1, 3}, 2); // Broadcasting по первой размерности
    mat2.print();

    // Умножение на скаляр
    Tensor<int> scaledMat = mat2 * 3;
    scaledMat.print();

    // Изменение формы (reshape)
    scaledMat.reshape({3, 2}); // Меняем форму на 3x2
    scaledMat.print();

    // Тензор на другом устройстве (пока только CPU, GPU — placeholder)
    Tensor<double> tensorGPU({2, 2}, 1.0, Device::GPU);
    tensorGPU.print();

    return 0;
}