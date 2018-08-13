#pragma once

enum GeometryType {
    GeometryTypeUnknown = 0,
    GeometryTypeSphere = 1,
};

namespace three {
class Geometry {
public:
    virtual GeometryType type();
};
}