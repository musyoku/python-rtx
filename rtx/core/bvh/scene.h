#pragma once
#include "../class/geometry.h"
#include "../header/glm.h"
#include "geometry.h"
#include <vector>
#include <memory>

namespace rtx {
namespace bvh {
    namespace scene {
        class Node {
        private:
            bool _is_leaf;
            std::vector<int> _assigned_object_ids;
            std::shared_ptr<geometry::GeometryBVH> _geometry_bvh;

        public:
            glm::vec3f _aabb_min;
            glm::vec3f _aabb_max;
            Node(std::vector<int> assigned_object_ids, std::vector<std::shared_ptr<Geometry>>& geometries);
            int _id;
            std::unique_ptr<Node> _left;
            std::unique_ptr<Node> _right;
            std::vector<int>& object_ids();
        };
        class SceneBVH {
        public:
            SceneBVH(std::vector<std::shared_ptr<Geometry>>& geometries);
            std::unique_ptr<Node> _root;
            void split(Node* parent);
        };
    }
}
}