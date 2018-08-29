#include "geometry.h"
#include "../class/geometry.h"

namespace rtx {
namespace bvh {
    namespace geometry {
        Node::Node(std::vector<glm::vec3i>& faces,
            std::vector<glm::vec3f>& vertices,
            glm::vec3f aabb_min,
            glm::vec3f aabb_max)
        {
            _node_index_base = 1;
            _faces = faces;
            _vertices = vertices;
            _aabb_min = aabb_min;
            _aabb_max = aabb_max;
            _id = _node_index_base;
            _node_index_base++;
        }
    }
}
}