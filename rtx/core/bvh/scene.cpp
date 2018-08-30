#include "scene.h"
#include <cassert>
#include <cfloat>
#include <iostream>
#include <utility>

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
                _aabb_max.x = geometry->_aabb_max.x > _aabb_max.x ? geometry->_aabb_max.x : _aabb_max.x;
                _aabb_max.y = geometry->_aabb_max.y > _aabb_max.y ? geometry->_aabb_max.y : _aabb_max.y;
                _aabb_max.z = geometry->_aabb_max.z > _aabb_max.z ? geometry->_aabb_max.z : _aabb_max.z;
                _aabb_min.x = geometry->_aabb_min.x < _aabb_min.x ? geometry->_aabb_min.x : _aabb_min.x;
                _aabb_min.y = geometry->_aabb_min.y < _aabb_min.y ? geometry->_aabb_min.y : _aabb_min.y;
                _aabb_min.z = geometry->_aabb_min.z < _aabb_min.z ? geometry->_aabb_min.z : _aabb_min.z;
            }

            glm::vec3f max_center = glm::vec3f(0.0f);
            glm::vec3f min_center = glm::vec3f(FLT_MAX);
            for (int object_id : assigned_object_indices) {
                auto& geometry = geometry_array.at(object_id);
                max_center.x = geometry->_center.x > max_center.x ? geometry->_center.x : max_center.x;
                max_center.y = geometry->_center.y > max_center.y ? geometry->_center.y : max_center.y;
                max_center.z = geometry->_center.z > max_center.z ? geometry->_center.z : max_center.z;
                min_center.x = geometry->_center.x < min_center.x ? geometry->_center.x : min_center.x;
                min_center.y = geometry->_center.y < min_center.y ? geometry->_center.y : min_center.y;
                min_center.z = geometry->_center.z < min_center.z ? geometry->_center.z : min_center.z;
                std::cout << geometry->type() << ": " << geometry->_center.x << ", " << geometry->_center.y << ", " << geometry->_center.z << std::endl;
            }
            glm::vec3f axis_length = max_center - min_center;
            std::cout << axis_length.x << ", " << axis_length.y << ", " << axis_length.z << std::endl;
        }
        SceneBVH::SceneBVH(std::vector<std::shared_ptr<Geometry>>& geometry_array)
        {
            std::vector<int> assigned_object_indices;
            for (int object_id = 0; object_id < (int)geometry_array.size(); object_id++) {
                assigned_object_indices.push_back(object_id);
            }
            _root = std::make_unique<Node>(assigned_object_indices, geometry_array);
        }
        rtx::array<float> SceneBVH::serialize()
        {
        }
    }
}
}