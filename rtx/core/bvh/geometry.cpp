#include "geometry.h"

namespace rtx {
namespace bvh {
    namespace geometry {
        static int geometry_bvh_node_current_id = 1;
        Node::Node(std::vector<glm::vec3i>& face_vertex_indices_array,
            std::vector<glm::vec3f>& vertex_array,
            glm::vec3f aabb_min,
            glm::vec3f aabb_max)
            : _face_vertex_indices_array(face_vertex_indices_array)
            , _vertex_array(vertex_array)
        {
            _aabb_min = aabb_min;
            _aabb_max = aabb_max;
            _id = geometry_bvh_node_current_id;
            geometry_bvh_node_current_id++;
        }
        GeometryBVH::GeometryBVH(std::vector<glm::vec3i>& face_vertex_indices_array,
            std::vector<glm::vec4f>& vertex_array)
            : _face_vertex_indices_array(face_vertex_indices_array)
            , _vertex_array(vertex_array)
        {
        }
    }
}
}