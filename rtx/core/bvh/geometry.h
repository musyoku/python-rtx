#pragma once
#include "../header/glm.h"
#include <memory>
#include <vector>

namespace rtx {
namespace bvh {
    namespace geometry {
        class Node {
        private:
            bool _is_leaf;
            std::vector<int> _assigned_face_indices;

        public:
            Node(std::vector<int> assigned_face_indices,
                std::vector<glm::vec3i>& face_vertex_indices_array,
                std::vector<glm::vec4f>& vertex_array,
                int num_split);
            int _id;
            std::unique_ptr<Node> _left;
            std::unique_ptr<Node> _right;
        };
        class GeometryBVH {
        private:
            std::unique_ptr<Node> _root;

        public:
            GeometryBVH(std::vector<glm::vec3i>& face_vertex_indices_array,
                std::vector<glm::vec4f>& vertex_array,
                int num_split);
            Node* root();
        };
    }
}
}