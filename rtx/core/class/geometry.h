#pragma once
#include "enum.h"

namespace rtx {
class Geometry {
public:
    virtual GeometryType type();
};
}