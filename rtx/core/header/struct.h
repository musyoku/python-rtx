#pragma once

typedef struct RTXVector3f {
    float x;
    float y;
    float z;
} RTXVector3f;

typedef struct RTXThreadedBVHNode {
    int hit_node_index;
    int miss_node_index;
    int assigned_face_index_start;
    int assigned_face_index_end;
    RTXVector3f aabb_max;
    RTXVector3f aabb_min;
} RTXThreadedBVHNode;

typedef struct RTXThreadedBVH {
    int num_nodes;
    int node_index_offset; // offset of the node from the start of the serialzied node array
} RTXThreadedBVH;

typedef RTXVector3f RTXGeometryVertex;

typedef struct RTXGeometryFace {
    int a;  // vertex index
    int b;  // vertex index
    int c;  // vertex index
} RTXGeometryFace;

typedef struct RTXRay {
    RTXVector3f direction;
    RTXVector3f origin;
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