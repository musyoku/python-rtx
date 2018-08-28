#pragma once

enum MaterialType {
    MaterialTypeUnknown = 0,
    MaterialTypeLambert = 1,
    MaterialTypeMetal = 2,
    MaterialTypeEmissive = 3,
};

enum GeometryType {
    GeometryTypeUnknown = 0,
    GeometryTypeStandard = 1,
    GeometryTypeSphere = 2,
    GeometryTypeBox = 3,
};

enum CameraType {
    CameraTypeUnknown = 0,
    CameraTypePerspective = 1,
    CameraTypeOrthogonal = 2,
};