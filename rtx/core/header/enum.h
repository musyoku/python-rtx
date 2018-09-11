#pragma once

enum RTXAxis {
    RTXAxisX = 1,
    RTXAxisY,
    RTXAxisZ,
};

enum RTXMaterialType {
    RTXMaterialTypeLambert = 1,
};

enum RTXObjectType {
    RTXObjectTypeStandardGeometry = 0b0010,
    RTXObjectTypeSphereGeometry = 0b0100,
    RTXObjectTypeRectAreaLight = 0b0001,
};

enum RTXCameraType {
    RTXCameraTypePerspective = 1,
    RTXCameraTypeOrthogonal,
};

#define SCENE_BVH_TERMINAL_NODE 255
#define SCENE_BVH_INNER_NODE 255