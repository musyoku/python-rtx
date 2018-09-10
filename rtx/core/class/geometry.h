#pragma once
#include "object.h"
#include <memory>

namespace rtx {
class Geometry : public Object {
    bool is_light() const override;
};
}