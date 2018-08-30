#include "scene.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <iostream>
#include <utility>

namespace rtx {
namespace bvh {
    namespace scene {
        int detect_longest_axis(const glm::vec3f& axis_length)
        {
            if (axis_length.x > axis_length.y) {
                if (axis_length.x > axis_length.z) {
                    return RTX_AXIS_X;
                }
                return RTX_AXIS_Z;
            }
            if (axis_length.y > axis_length.x) {
                if (axis_length.y > axis_length.z) {
                    return RTX_AXIS_Y;
                }
                return RTX_AXIS_Z;
            }
            if (axis_length.x > axis_length.z) {
                if (axis_length.y > axis_length.x) {
                    return RTX_AXIS_Y;
                }
                return RTX_AXIS_X;
            }
            return RTX_AXIS_Z;
        }
        bool compare_position(const std::pair<int, float>& a, const std::pair<int, float>& b)
        {
            return a.second < b.second;
        }
        static int scene_bvh_node_current_id = 1;
        Node::Node(std::vector<int> assigned_object_indices,
            std::vector<std::shared_ptr<Geometry>>& geometry_array)
        {
            std::cout << "===================================" << std::endl;
            assert(assigned_object_indices.size() > 0);
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

            if (assigned_object_indices.size() == 1) {
                return;
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
            const glm::vec3f axis_length = max_center - min_center;
            int longest_axis = detect_longest_axis(axis_length);
            std::cout << "longest: " << longest_axis << std::endl;
            std::vector<std::pair<int, float>> object_center_array;
            for (int object_id : assigned_object_indices) {
                auto& geometry = geometry_array.at(object_id);
                if (longest_axis == RTX_AXIS_X) {
                    object_center_array.emplace_back(object_id, geometry->_center.x);
                } else if (longest_axis == RTX_AXIS_Y) {
                    object_center_array.emplace_back(object_id, geometry->_center.y);
                } else {
                    object_center_array.emplace_back(object_id, geometry->_center.z);
                }
            }
            std::sort(object_center_array.begin(), object_center_array.end(), compare_position);
            std::cout << "sort:" << std::endl;
            for (auto& pair : object_center_array) {
                std::cout << pair.first << ": " << pair.second << std::endl;
            }
            int split_index = object_center_array.size() / 2;
            std::cout << "split: " << split_index << std::endl;
            std::vector<int> left_assigned_indices;
            std::vector<int> right_assigned_indices;
            for (int n = 0; n < split_index; n++) {
                auto& pair = object_center_array.at(n);
                left_assigned_indices.push_back(pair.first);
            }
            for (int n = split_index; n < (int)assigned_object_indices.size(); n++) {
                auto& pair = object_center_array.at(n);
                right_assigned_indices.push_back(pair.first);
            }
            std::cout << "left:" << std::endl;
            for (auto index : left_assigned_indices) {
                std::cout << index << std::endl;
            }
            std::cout << "right:" << std::endl;
            for (auto index : right_assigned_indices) {
                std::cout << index << std::endl;
            }
            _left = std::make_unique<Node>(left_assigned_indices, geometry_array);
            _right = std::make_unique<Node>(right_assigned_indices, geometry_array);
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