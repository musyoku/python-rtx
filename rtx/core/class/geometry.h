#pragma once

enum GeometryType {
    GeometryTypeUnknown = 0,
    GeometryTypeStandard = 1,
    GeometryTypeSphere = 2,
};

namespace rtx {
class Geometry {
public:
    virtual GeometryType type();
};
}