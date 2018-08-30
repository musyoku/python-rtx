#include "scene.h"
#include <cassert>
#include <iostream>

namespace rtx {
namespace bvh {
    namespace scene {
        static int scene_bvh_node_current_id = 1;
        Node::Node(std::vector<int> assigned_object_indices,
            std::vector<std::shared_ptr<Geometry>>& geometry_array)
        {
            assert(assigned_object_indices.size() <= geometry_array.size());
            _assigned_object_indices = assigned_object_indices;
            _id = scene_bvh_node_current_id;
            scene_bvh_node_current_id++;
            _aabb_max = glm::vec4f(0.0f);
            _aabb_min = glm::vec4f(0.0f);
            for (int object_id : assigned_object_indices) {
                auto& geometry = geometry_array.at(object_id);
                auto bvh = geometry->bvh();
                if (bvh) {
                    auto root = geometry->bvh()->root();
                } else {
                }
            }
            _aabb_max /= float(assigned_object_indices.size());
            _aabb_min /= float(assigned_object_indices.size());
            std::cout << "AAAB(min): " << _aabb_min.x << ", " << _aabb_min.y << ", " << _aabb_min.z << std::endl;
            std::cout << "AAAB(max): " << _aabb_max.x << ", " << _aabb_max.y << ", " << _aabb_max.z << std::endl;
        }
        SceneBVH::SceneBVH(std::vector<std::shared_ptr<Geometry>>& geometry_array)
        {
            std::vector<int> assigned_object_indices;
            for (int object_id = 0; object_id < (int)geometry_array.size(); object_id++) {
                assigned_object_indices.push_back(object_id);
            }
            _root = std::make_unique<Node>(assigned_object_indices, geometry_array);
        }
        void SceneBVH::split(const std::unique_ptr<Node>& parent)
        {
        }
        rtx::array<float> SceneBVH::serialize()
        {
        }
    }
}
}