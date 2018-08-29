#include "scene.h"

namespace rtx {
namespace bvh {
    namespace scene {
        static int scene_bvh_node_current_id = 1;
        Node::Node(std::vector<int> object_ids, glm::vec3f aabb_min, glm::vec3f aabb_max)
        {
            _object_ids = object_ids;
            _aabb_min = aabb_min;
            _aabb_max = aabb_max;
            _id = scene_bvh_node_current_id;
            scene_bvh_node_current_id++;
        }
    }
}
}