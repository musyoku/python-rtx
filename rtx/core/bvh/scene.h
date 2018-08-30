#pragma once
#include "../class/geometry.h"
#include "../header/array.h"
#include "../header/glm.h"
#include "geometry.h"
#include <memory>
#include <vector>

namespace rtx {
namespace bvh {
    namespace scene {
        int detect_longest_axis(const glm::vec3f& axis_length);
        bool compare_position(const std::pair<int, float>& a, const std::pair<int, float>& b);
        class Node {
        public:
            bool _is_leaf;
            unsigned int _index;
            std::vector<int> _assigned_object_indices;
            std::shared_ptr<geometry::GeometryBVH> _geometry_bvh;
            glm::vec4f _aabb_min;
            glm::vec4f _aabb_max;
            Node(std::vector<int> assigned_object_indices, std::vector<std::shared_ptr<Geometry>>& geometry_array);
            std::shared_ptr<Node> _left;
            std::shared_ptr<Node> _right;
            std::shared_ptr<Node> _miss;
            std::shared_ptr<Node> _hit;
            std::vector<int>& object_ids();
            int num_children();
            void set_hit_and_miss_links();
            void collect_children(std::vector<std::shared_ptr<Node>>& children);
        };
        class SceneBVH {
        public:
            SceneBVH(std::vector<std::shared_ptr<Geometry>>& geometry_array);
            std::shared_ptr<Node> _root;
            int num_nodes();
            void serialize(rtx::array<unsigned int>& buffer);
        };
    }
}
}