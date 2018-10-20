#pragma once
#include "enum.h"

typedef struct rtxVector4f {
    float x;
    float y;
    float z;
    float w;
} rtxVector4f;

typedef struct rtxRGBAColor {
    float r;
    float g;
    float b;
    float a;
} rtxRGBAColor;

typedef struct rtxThreadedBVHNode {
    int hit_node_index;
    int miss_node_index;
    int assigned_face_index_start;
    int assigned_face_index_end;
    rtxVector4f aabb_max;
    rtxVector4f aabb_min;
} rtxThreadedBVHNode;

typedef struct rtxThreadedBVH {
    int num_nodes;
    int serial_node_index_offset; // offset of the node from the start of the serialzied node array
} rtxThreadedBVH;

typedef rtxVector4f rtxVertex;

typedef struct rtxFaceVertexIndex {
    int a;
    int b;
    int c;
    int w; // dammy for float4
} rtxFaceVertexIndex;

typedef struct rtxUVCoordinate {
    float u;
    float v;
} rtxUVCoordinate;

typedef struct rtxRay {
    rtxVector4f direction;
    rtxVector4f origin;
} rtxRay;

typedef struct rtxRGBAPixel {
    float r;
    float g;
    float b;
    float a;
} rtxRGBAPixel;

typedef struct rtxLayeredMaterialTypes {
    int outside;
    int middle;
    int inside;
} rtxLayeredMaterialTypes;

typedef struct rtxObject {
    int num_faces;
    int serialized_face_index_offset; // offset of the face from the start of the serialzied face array
    int num_vertices;
    int serialized_vertex_index_offset; // offset of the vertex from the start of the serialzied face array
    int geometry_type;
    int num_material_layers;
    rtxLayeredMaterialTypes layerd_material_types;
    int material_attribute_byte_array_offset;
    int serialized_uv_coordinates_offset;
    int mapping_type;
    int mapping_index;
} rtxObject;

typedef char RTXMappingAttributeByte;
typedef char rtxMaterialAttributeByte;

typedef struct rtxEmissiveMaterialAttribute {
    float intensity;
    bool visible;
} rtxEmissiveMaterialAttribute;

typedef struct rtxLambertMaterialAttribute {
    float albedo;
} rtxLambertMaterialAttribute;

typedef struct rtxOrenNayarMaterialAttribute {
    float albedo;
    float roughness;
} rtxOrenNayarMaterialAttribute;

typedef struct rtxMCRTKernelArguments {
    int num_active_texture_units;
    int num_rays_per_thread;
    int num_rays_per_pixel;
    int max_bounce;
    float ray_origin_z;
    int screen_width;
    int screen_height;
    RTXCameraType camera_type;
    rtxRGBAColor ambient_color;
    int face_vertex_index_array_size;
    int vertex_array_size;
    int object_array_size;
    int material_attribute_byte_array_size;
    int threaded_bvh_array_size;
    int color_mapping_array_size;
    int threaded_bvh_node_array_size;
    int uv_coordinate_array_size;
    int curand_seed;
    bool supersampling_enabled;
} rtxMCRTKernelArguments;

typedef struct rtxNEEKernelArguments {
    int num_active_texture_units;
    int num_rays_per_thread;
    int num_rays_per_pixel;
    int max_bounce;
    float ray_origin_z;
    int screen_width;
    int screen_height;
    RTXCameraType camera_type;
    rtxRGBAColor ambient_color;
    int face_vertex_index_array_size;
    int vertex_array_size;
    int object_array_size;
    int material_attribute_byte_array_size;
    int threaded_bvh_array_size;
    int color_mapping_array_size;
    int threaded_bvh_node_array_size;
    int light_sampling_table_size;
    int uv_coordinate_array_size;
    float total_light_face_area;
    int curand_seed;
    bool supersampling_enabled;
} rtxNEEKernelArguments;