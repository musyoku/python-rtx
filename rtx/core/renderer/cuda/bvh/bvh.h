#pragma once
#include "../../../geometry/standard.h"
#include "../../../header/array.h"
#include "../../../header/glm.h"
#include "../header/ray_tracing.h"
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
        Node(std::vector<int> assigned_face_indices,
            std::shared_ptr<StandardGeometry>& geometry,
            int& current_index);
        std::shared_ptr<Node> _left;
        std::shared_ptr<Node> _right;
        std::shared_ptr<Node> _miss;
        std::shared_ptr<Node> _hit;
        int num_children();
        void set_hit_and_miss_links();
        void collect_children(std::vector<std::shared_ptr<Node>>& children);
    };
}
class BVH {
private:
    int _node_current_index;
    int _num_nodes;

public:
    BVH(std::shared_ptr<StandardGeometry>& geometry);
    std::shared_ptr<bvh::Node> _root;
    int num_nodes();
    void serialize(rtx::array<RTXThreadedBVHNode>& node_array, int offset);
};
}