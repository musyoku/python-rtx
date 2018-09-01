#include "scene.h"
#include <algorithm>
#include <bitset>
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
        Node::Node(std::vector<int> assigned_object_indices,
            std::vector<std::shared_ptr<Geometry>>& geometry_array,
            int& current_index)
        {
            // std::cout << "===================================" << std::endl;
            assert(assigned_object_indices.size() > 0);
            assert(assigned_object_indices.size() <= geometry_array.size());
            _assigned_object_indices = assigned_object_indices;
            _index = current_index;
            // std::cout << "id: " << _index << std::endl;
            current_index++;
            if (current_index > 253) {
                throw std::runtime_error("BVH Error: Too many nodes.");
            }
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
                _is_leaf = true;
                auto& geometry = geometry_array.at(assigned_object_indices.at(0));
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
                // std::cout << geometry->type() << ": " << geometry->_center.x << ", " << geometry->_center.y << ", " << geometry->_center.z << std::endl;
            }
            const glm::vec3f axis_length = max_center - min_center;
            int longest_axis = detect_longest_axis(axis_length);
            // std::cout << "longest: " << longest_axis << std::endl;
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
            // std::cout << "sort:" << std::endl;
            // for (auto& pair : object_center_array) {
            //     std::cout << pair.first << ": " << pair.second << std::endl;
            // }
            int split_index = object_center_array.size() / 2;
            // std::cout << "split: " << split_index << std::endl;
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
            // std::cout << "left:" << std::endl;
            // for (auto index : left_assigned_indices) {
            //     std::cout << index << std::endl;
            // }
            // std::cout << "right:" << std::endl;
            // for (auto index : right_assigned_indices) {
            //     std::cout << index << std::endl;
            // }
            _left = std::make_shared<Node>(left_assigned_indices, geometry_array, current_index);
            _right = std::make_shared<Node>(right_assigned_indices, geometry_array, current_index);
        }
        void Node::set_hit_and_miss_links()
        {
            if (_left) {
                _hit = _left;
                if (_right) {
                    _left->_miss = _right;
                } else {
                    _left->_miss = _miss;
                }
                _left->set_hit_and_miss_links();
            }
            if (_right) {
                _right->_miss = _miss;
                _right->set_hit_and_miss_links();
            }
        }
        int Node::num_children()
        {
            int num_children = 0;
            if (_left) {
                num_children += _left->num_children() + 1;
            }
            if (_right) {
                num_children += _right->num_children() + 1;
            }
            return num_children;
        }
        void Node::collect_children(std::vector<std::shared_ptr<Node>>& children)
        {
            if (_left) {
                children.push_back(_left);
                _left->collect_children(children);
            }
            if (_right) {
                children.push_back(_right);
                _right->collect_children(children);
            }
        }
        SceneBVH::SceneBVH(std::vector<std::shared_ptr<Geometry>>& geometry_array)
        {
            std::vector<int> assigned_object_indices;
            for (int object_id = 0; object_id < (int)geometry_array.size(); object_id++) {
                assigned_object_indices.push_back(object_id);
            }
            _node_current_index = 0;
            _root = std::make_shared<Node>(assigned_object_indices, geometry_array, _node_current_index);
            _root->set_hit_and_miss_links();
        }
        int SceneBVH::num_nodes()
        {
            int num_nodes = _root->num_children() + 1;
            // std::cout << "#nodes: " << num_nodes << std::endl;
            return num_nodes;
        }
        void SceneBVH::serialize(rtx::array<unsigned int>& node_buffer, rtx::array<float>& aabb_buffer)
        {
            // std::cout << "serialize:" << std::endl;
            int num_nodes = this->num_nodes();
            assert(node_buffer.size() == num_nodes);
            std::vector<std::shared_ptr<Node>> children = { _root };
            _root->collect_children(children);
            for (auto& node : children) {
                unsigned int hit_bit = node->_hit ? node->_hit->_index : 255;
                unsigned int miss_bit = node->_miss ? node->_miss->_index : 255;
                unsigned int object_id_bit = node->_is_leaf ? node->_assigned_object_indices[0] : 255;
                unsigned int binary_path = (hit_bit << 16) + (miss_bit << 8) + object_id_bit;
                node_buffer[node->_index] = binary_path;
                aabb_buffer[node->_index * 8 + 0] = node->_aabb_max.x;
                aabb_buffer[node->_index * 8 + 1] = node->_aabb_max.y;
                aabb_buffer[node->_index * 8 + 2] = node->_aabb_max.z;
                aabb_buffer[node->_index * 8 + 3] = 1.0f;
                aabb_buffer[node->_index * 8 + 4] = node->_aabb_min.x;
                aabb_buffer[node->_index * 8 + 5] = node->_aabb_min.y;
                aabb_buffer[node->_index * 8 + 6] = node->_aabb_min.z;
                aabb_buffer[node->_index * 8 + 7] = 1.0f;
                // std::cout << " index: ";
                // std::cout << node->_index;
                // std::cout << " left: ";
                // if (node->_left) {
                //     std::cout << node->_left->_index;
                // }
                // std::cout << " right: ";
                // if (node->_right) {
                //     std::cout << node->_right->_index;
                // }
                // std::cout << " hit: ";
                // if (node->_hit) {
                //     std::cout << node->_hit->_index;
                // }
                // std::cout << " miss: ";
                // if (node->_miss) {
                //     std::cout << node->_miss->_index;
                // }
                // std::cout << " object: ";
                // if (node->_assigned_object_indices.size() == 1) {
                //     std::cout << node->_assigned_object_indices[0];
                // }
                // std::cout << " binary: " << binary_path;
                // std::cout << std::endl;
            }
        }
    }
}
}