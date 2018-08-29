#include "scene.h"

namespace rtx {
namespace bvh {
    namespace scene {
        static int scene_bvh_node_current_id = 1;
        Node::Node(std::vector<int> assigned_object_ids,
            std::vector<std::shared_ptr<Geometry>>& geometries)
        {
            _assigned_object_ids = assigned_object_ids;
            _id = scene_bvh_node_current_id;
            scene_bvh_node_current_id++;
        }
        SceneBVH::SceneBVH(std::vector<std::shared_ptr<Geometry>>& geometries)
        {
            std::vector<int> assigned_object_ids;
            for (int object_id = 0; object_id < (int)geometries.size(); object_id++) {
                assigned_object_ids.push_back(object_id);
            }
            _root = std::make_unique<Node>(assigned_object_ids, geometries);
        }
    }
}
}