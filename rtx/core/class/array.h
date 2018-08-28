#pragma once
#include <cassert>

namespace rtx {
template <typename T>
class array {
private:
    T* _array;
    int _size;

public:
    array()
    {
        _array = nullptr;
        _size = 0;
    }
    array(int size)
    {
        _array = new T[size];
        _size = size;
    }
    array(const array& a)
    {
        _size = a._size;
        if (_array != nullptr) {
            delete[] _array;
        }
        _array = new T[_size];
        for (int i = 0; i < _size; i++) {
            _array[i] = a._array[i];
        }
    }
    ~array()
    {
        if (_array != nullptr) {
            delete[] _array;
        }
    }
    void fill(T value)
    {
        for (int i = 0; i < _size; i++) {
            _array[i] = value;
        }
    }
    array& operator=(const array& a)
    {
        if (_array != nullptr) {
            delete[] _array;
        }
        _size = a._size;
        _array = new T[_size];
        for (int i = 0; i < _size; i++) {
            _array[i] = a._array[i];
        }
        return *this;
    }
    T& operator[](int i)
    {
        assert(i < _size);
        return _array[i];
    }
    const T& operator[](int i) const
    {
        assert(i < _size);
        return _array[i];
    }
    int size() const
    {
        return _size;
    }
};
}