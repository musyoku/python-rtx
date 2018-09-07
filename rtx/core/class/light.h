#pragma once
#include "object.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace rtx {
class Light : public Object {
protected:
    float _brightness;

public:
    float brightness() const;
};
}