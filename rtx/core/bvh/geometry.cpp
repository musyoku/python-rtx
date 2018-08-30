#include "geometry.h"
#include <cassert>
#include <cfloat>
#include <iostream>

namespace rtx {
namespace bvh {
    namespace geometry {
        static int geometry_bvh_node_current_id = 1;
        Node::Node(std::vector<int> assigned_face_indices,
            std::vector<glm::vec3i>& face_vertex_indices_array,
            std::vector<glm::vec4f>& vertex_array,
            int num_split)
            : _face_vertex_indices_array(face_vertex_indices_array)
            , _vertex_array(vertex_array)
        {
            assert(num_split > 0);

            _id = geometry_bvh_node_current_id;
            geometry_bvh_node_current_id++;
            _assigned_face_indices = assigned_face_indices;
            _is_leaf = false;

            float min_x = FLT_MAX, max_x = 0.0f;
            float min_y = FLT_MAX, max_y = 0.0f;
            float min_z = FLT_MAX, max_z = 0.0f;
            for (int face_index : _assigned_face_indices) {
                glm::vec3i& face = face_vertex_indices_array[face_index];
                glm::vec4f& va = vertex_array[face[0]];
                glm::vec4f& vb = vertex_array[face[1]];
                glm::vec4f& vc = vertex_array[face[2]];
                min_x = (va.x < min_x) ? va.x : min_x;
                min_y = (vb.y < min_y) ? vb.y : min_y;
                min_z = (vc.z < min_z) ? vc.z : min_z;
                max_x = (va.x > max_x) ? va.x : max_x;
                max_y = (vb.y > max_y) ? vb.y : max_y;
                max_z = (vc.z > max_z) ? vc.z : max_z;
            }
            float length_x = max_x - min_x;
            float length_y = max_y - min_y;
            float length_z = max_z - min_z;
            std::cout << "length_x: " << length_x << ", "
                      << "length_y: " << length_y << ", "
                      << "length_z: " << length_z << std::endl;

            if (num_split == 1) {
                // _aabb_min = glm::vec4f(min_x, min_y, min_z, 1.0f);
                // _aabb_max = glm::vec4f(max_x, max_y, max_z, 1.0f);
                _is_leaf = true;
                return;
            }
        }
        GeometryBVH::GeometryBVH(std::vector<glm::vec3i>& face_vertex_indices_array,
            std::vector<glm::vec4f>& vertex_array,
            int num_split)
            : _face_vertex_indices_array(face_vertex_indices_array)
            , _vertex_array(vertex_array)
        {
            std::vector<int> face_indices;
            for (unsigned int face_index = 0; face_index < face_vertex_indices_array.size(); face_index++) {
                face_indices.push_back(face_index);
            }
            _root = std::make_unique<Node>(face_indices,
                face_vertex_indices_array,
                vertex_array,
                num_split);
        }
        Node* GeometryBVH::root()
        {
            assert(!!_root == true);
            return _root.get();
        }
    }
}
}