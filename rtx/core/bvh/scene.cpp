#include "scene.h"

namespace rtx {
namespace bvh {
    namespace scene {
        Node::Node(std::vector<int> object_ids, glm::vec3f aabb_min, glm::vec3f aabb_max)
        {
            _node_index_base = 1;
            _object_ids = object_ids;
            _aabb_min = aabb_min;
            _aabb_max = aabb_max;
            _id = _node_index_base;
            _node_index_base++;
        }
    }
}
}