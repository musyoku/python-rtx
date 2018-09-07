#pragma once

typedef struct RTXVector4f {
    float x;
    float y;
    float z;
    float w;
} RTXVector4f;

typedef struct RTXThreadedBVHNode {
    int hit_node_index;
    int miss_node_index;
    int assigned_face_index_start;
    int assigned_face_index_end;
    RTXVector4f aabb_max;
    RTXVector4f aabb_min;
} RTXThreadedBVHNode;

typedef struct RTXThreadedBVH {
    int num_nodes;
    int node_index_offset; // offset of the node from the start of the serialzied node array
} RTXThreadedBVH;

typedef RTXVector4f RTXVertex;

typedef struct RTXFace {
    int a;  // vertex index
    int b;  // vertex index
    int c;  // vertex index
    int dummy_axis;
} RTXFace;

typedef struct RTXRay {
    RTXVector4f direction;
    RTXVector4f origin;
} RTXRay;

typedef struct RTXPixel {
    float r;
    float g;
    float b;
} RTXPixel;

typedef struct RTXObject {
    int num_faces;
    int face_index_offset;      // offset of the face from the start of the serialzied face array
    int num_vertices;
    int vertex_index_offset;    // offset of the vertex from the start of the serialzied face array
    bool bvh_enabled;
    int bvh_index;
} RTXObject;