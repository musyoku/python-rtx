#pragma once
#include <cassert>
#include <execinfo.h>
#include <iostream>
#include <unistd.h>

namespace rtx {
template <typename T>
class array {
private:
    T* _array;
    int _size;
    array(const array& a)
    {
        _size = a._size;
        if (_array != nullptr) {
            delete[] _array;
        }
        _array = new T[_size];
        for (int index = 0; index < _size; index++) {
            _array[index] = a._array[index];
        }
    }

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
    ~array()
    {
        if (_array != nullptr) {
            delete[] _array;
            _array = nullptr;
            _size = 0;
        }
    }
    void fill(T value)
    {
        for (int index = 0; index < _size; index++) {
            _array[index] = value;
        }
    }
    array& operator=(const array& a)
    {
        if (_array != nullptr) {
            delete[] _array;
        }
        _size = a._size;
        _array = new T[_size];
        for (int index = 0; index < _size; index++) {
            _array[index] = a._array[index];
        }
        return *this;
    }
    T& operator[](int index)
    {
        if (index >= _size) {
            void* bt[100];
            int n = backtrace(bt, 100);
            backtrace_symbols_fd(bt, n, STDERR_FILENO);
        }
        assert(index < _size);
        assert(_array != nullptr);
        return _array[index];
    }
    const T& operator[](int index) const
    {
        if (index >= _size) {
            void* bt[100];
            int n = backtrace(bt, 100);
            backtrace_symbols_fd(bt, n, STDERR_FILENO);
        }
        assert(index < _size);
        assert(_array != nullptr);
        return _array[index];
    }
    int size() const
    {
        return _size;
    }
    size_t bytes() const
    {
        return sizeof(T) * _size;
    }
    T* data()
    {
        return _array;
    }
};
}