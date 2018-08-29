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
            std::vector<int> _assigned_face_ids;
            std::vector<glm::vec3i>& _face_vertex_indices_array;
            std::vector<glm::vec3f>& _vertices;
            static int _node_index_base;

        public:
            glm::vec3f _aabb_min;
            glm::vec3f _aabb_max;
            Node(std::vector<glm::vec3i>& faces,
                std::vector<glm::vec3f>& vertices,
                glm::vec3f aabb_min,
                glm::vec3f aabb_max);
            int _id;
            Node* _left;
            Node* _right;
        };
        class GeometryBVH {
        private:
            std::vector<glm::vec3i>& _face_vertex_indices_array;
            std::vector<glm::vec4f>& _vertex_array;

        public:
            GeometryBVH(std::vector<glm::vec3i>& _face_vertex_indices_array,
                std::vector<glm::vec4f>& _vertex_array);
            Node* _root;
            void split(Node* parent);
        };
    }
}
}