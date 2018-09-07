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
    RTXObjectTypeStandardGeometry = 1,
    RTXObjectTypeSphereGeometry,
    RTXObjectTypeRectAreaLight,
};

enum RTXCameraType {
    RTXCameraTypePerspective = 1,
    RTXCameraTypeOrthogonal,
};

#define SCENE_BVH_TERMINAL_NODE 255
#define SCENE_BVH_INNER_NODE 255