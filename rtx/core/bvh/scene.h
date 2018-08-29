#pragma once
#include "../class/geometry.h"
#include "../header/glm.h"
#include <vector>

namespace rtx {
namespace bvh {
    namespace scene {
        class Node {
        private:
            bool _is_leaf;
            std::vector<int> _object_ids;
            static int _node_index_base;

        public:
            glm::vec3f _aabb_min;
            glm::vec3f _aabb_max;
            Node(std::vector<int> object_ids, glm::vec3f aabb_min, glm::vec3f aabb_max);
            int _id;
            Node* _left;
            Node* _right;
            std::vector<int>& object_ids();
        };
        class SceneBVH {
        private:
            std::vector<Geometry*> _geometries;

        public:
            SceneBVH(std::vector<Geometry*>& geometries);
            Node* _root;
            void split(Node* parent);
        };
    }
}
}