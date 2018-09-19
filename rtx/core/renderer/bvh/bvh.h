#pragma once
#include "../../geometry/sphere.h"
#include "../../geometry/standard.h"
#include "../../header/array.h"
#include "../../header/glm.h"
#include <memory>
#include <vector>

namespace rtx {
namespace bvh {
    class Node {
    public:
        bool _is_leaf;
        unsigned int _index;
        std::vector<int> _assigned_face_indices;
        glm::vec3f _aabb_min;
        glm::vec3f _aabb_max;
        int _assigned_face_index_start;
        int _assigned_face_index_end;
        Node(std::vector<int> assigned_face_indices,
            std::shared_ptr<StandardGeometry>& geometry,
            int& current_node_index,
            int& current_assigned_face_index_offset);
        Node(std::vector<int> assigned_face_indices,
            std::shared_ptr<SphereGeometry>& geometry);
        std::shared_ptr<Node> _left;
        std::shared_ptr<Node> _right;
        std::shared_ptr<Node> _miss;
        std::shared_ptr<Node> _hit;
        int num_children();
        void set_hit_and_miss_links();
        void collect_children(std::vector<std::shared_ptr<Node>>& children);
        void collect_leaves(std::vector<std::shared_ptr<bvh::Node>>& leaves);
    };
}
class BVH {
private:
    int _current_node_index;
    int _current_assigned_face_index_offset;
    int _num_nodes;
    std::weak_ptr<Geometry> _geometry;

public:
    BVH(std::shared_ptr<Geometry>& geometry);
    std::shared_ptr<bvh::Node> _root;
    int num_nodes();
    void serialize_nodes(rtx::array<rtxThreadedBVHNode>& node_array, int serialization_offset);
    void serialize_faces(rtx::array<rtxFaceVertexIndex>& buffer, int serialization_offset);
    void collect_leaves(std::vector<std::shared_ptr<bvh::Node>>& leaves);
};
}