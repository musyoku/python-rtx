#pragma once

enum RTXAxis {
    RTXAxisX = 1,
    RTXAxisY,
    RTXAxisZ,
};

enum RTXMaterialType {
    RTXMaterialTypeLambert = 1,
    RTXMaterialTypeOrenNayar,
    RTXMaterialTypeSpecular,
    RTXMaterialTypeRefractive,
    RTXMaterialTypeEmissive,
};

enum RTXGeometryType {
    RTXGeometryTypeStandard = 1,
    RTXGeometryTypeSphere,
    RTXGeometryTypeCone,
    RTXGeometryTypeCylinder,
};

enum RTXMappingType {
    RTXMappingTypeSolidColor = 1,
    RTXMappingTypeTexture,
};

enum RTXCameraType {
    RTXCameraTypePerspective = 1,
    RTXCameraTypeOrthogonal,
};

#define SCENE_BVH_TERMINAL_NODE 255
#define SCENE_BVH_INNER_NODE 255