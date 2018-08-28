#pragma once
#include "standard.h"

namespace rtx {
class BoxGeometry : public StandardGeometry {
public:
    BoxGeometry(float width, float height, float depth);
};
}