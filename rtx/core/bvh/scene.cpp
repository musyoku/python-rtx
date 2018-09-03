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
        glm::vec3f merge_aabb_max(const glm::vec3f a, const glm::vec3f b)
        {
            glm::vec3f max;
            max.x = a.x > b.x ? a.x : b.x;
            max.y = a.y > b.y ? a.y : b.y;
            max.z = a.z > b.z ? a.z : b.z;
            return max;
        }
        glm::vec3f merge_aabb_min(const glm::vec3f a, const glm::vec3f b)
        {
            glm::vec3f min;
            min.x = a.x < b.x ? a.x : b.x;
            min.y = a.y < b.y ? a.y : b.y;
            min.z = a.z < b.z ? a.z : b.z;
            return min;
        }
        float compute_surface_area(const glm::vec3f max, const glm::vec3f min)
        {
            float dx = max.x - min.x;
            float dy = max.y - min.y;
            float dz = max.z - min.z;
            return 2 * (dx * dy + dx * dz + dy * dz);
        }

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
            _is_leaf = false;
            // std::cout << "id: " << _index << std::endl;
            current_index++;
            if (current_index > 253) {
                throw std::runtime_error("BVH Error: Too many objects.");
            }
            _aabb_max = glm::vec4f(-FLT_MAX);
            _aabb_min = glm::vec4f(FLT_MAX);

            for (int object_index : assigned_object_indices) {
                auto& geometry = geometry_array.at(object_index);
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

            glm::vec3f max_center = glm::vec3f(-FLT_MAX);
            glm::vec3f min_center = glm::vec3f(FLT_MAX);
            for (int object_index : assigned_object_indices) {
                auto& geometry = geometry_array.at(object_index);
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
            // if (_index == 0) {
            //     longest_axis = RTX_AXIS_X;
            // }

            // std::cout << "longest: " << longest_axis << std::endl;
            std::vector<std::pair<int, float>> object_center_array;
            for (int object_index : assigned_object_indices) {
                auto& geometry = geometry_array.at(object_index);
                if (longest_axis == RTX_AXIS_X) {
                    object_center_array.emplace_back(object_index, geometry->_center.x);
                } else if (longest_axis == RTX_AXIS_Y) {
                    object_center_array.emplace_back(object_index, geometry->_center.y);
                } else {
                    object_center_array.emplace_back(object_index, geometry->_center.z);
                }
            }
            std::sort(object_center_array.begin(), object_center_array.end(), compare_position);
            std::cout << "sort:" << std::endl;
            for (auto& pair : object_center_array) {
                std::cout << pair.first << ": " << pair.second << std::endl;
            }
            float whole_surface_area = compute_surface_area(_aabb_max, _aabb_min);
            std::cout << "whole_surface_area: " << whole_surface_area << std::endl;

            glm::vec3f volume_a_max(FLT_MAX);
            glm::vec3f volume_a_min(FLT_MAX);
            glm::vec3f volume_b_max(FLT_MAX);
            glm::vec3f volume_b_min(FLT_MAX);
            std::cout << "==============================================================" << std::endl;
            
            float min_cost = FLT_MAX;
            int min_cost_split_index = 0;
            for (int split_index = 1; split_index <= object_center_array.size() - 1; split_index++) {
                int volume_a_num_faces = 0;
                int volume_b_num_faces = 0;
                for (int position = 0; position < split_index; position++) {
                    int object_index = object_center_array[position].first;
                    auto& geometry = geometry_array.at(object_index);
                    glm::vec3f max = geometry->_aabb_max;
                    glm::vec3f min = geometry->_aabb_min;
                    // std::cout << "(left) object: " << object_index << ", max: " << max.x << ", " << max.y << ", " << max.z << " min: " << min.x << ", " << min.y << ", " << min.z << std::endl;
                    // std::cout << "      volume max: " << volume_a_max.x << ", " << volume_a_max.y << ", " << volume_a_max.z << ", min: " << volume_b_min.x << ", " << volume_b_min.y << ", " << volume_b_min.z << std::endl;
                    if (position == 0) {
                        volume_a_max = max;
                        volume_a_min = min;
                    } else {
                        volume_a_max = merge_aabb_max(volume_a_max, max);
                        volume_a_min = merge_aabb_min(volume_a_min, min);
                        // std::cout << "      merge: " << volume_a_max.x << ", " << volume_a_max.y << ", " << volume_a_max.z << ", min: " << volume_b_min.x << ", " << volume_b_min.y << ", " << volume_b_min.z << std::endl;
                    }
                    volume_a_num_faces += geometry->num_faces();
                }
                for (int position = split_index; position < object_center_array.size(); position++) {
                    int object_index = object_center_array[position].first;
                    auto& geometry = geometry_array.at(object_index);
                    glm::vec3f max = geometry->_aabb_max;
                    glm::vec3f min = geometry->_aabb_min;
                    // std::cout << "(right) object: " << object_index << ", max: " << max.x << ", " << max.y << ", " << max.z << " min: " << min.x << ", " << min.y << ", " << min.z << std::endl;
                    // std::cout << "      volume max: " << volume_a_max.x << ", " << volume_a_max.y << ", " << volume_a_max.z << ", min: " << volume_b_min.x << ", " << volume_b_min.y << ", " << volume_b_min.z << std::endl;
                    if (position == split_index) {
                        volume_b_max = max;
                        volume_b_min = min;
                    } else {
                        volume_b_max = merge_aabb_max(volume_b_max, max);
                        volume_b_min = merge_aabb_min(volume_b_min, min);
                        // std::cout << "      merge: " << volume_b_max.x << ", " << volume_b_max.y << ", " << volume_b_max.z << ", min: " << volume_b_min.x << ", " << volume_b_min.y << ", " << volume_b_min.z << std::endl;
                    }
                    volume_b_num_faces += geometry->num_faces();
                }
                float surface_a = compute_surface_area(volume_a_max, volume_a_min);
                float surface_b = compute_surface_area(volume_b_max, volume_b_min);
                std::cout << "split: " << split_index << ", surface_a: " << surface_a << ", surface_b: " << surface_b << std::endl;
                std::cout << "split: " << split_index << ", faces_a: " << volume_a_num_faces << ", faces_b: " << volume_b_num_faces << std::endl;
                std::cout << "split: " << split_index << ", a: " << (surface_a * volume_a_num_faces) << ", b: " << (surface_b * volume_b_num_faces) << std::endl;
                float cost = surface_a * volume_a_num_faces + surface_b * volume_b_num_faces;
                if(cost < min_cost){
                    min_cost = cost;
                    min_cost_split_index = split_index;
                }
            }
            std::cout << "min_cost: " << min_cost << std::endl;
            std::cout << "min_cost_split_index: " << min_cost_split_index << std::endl;
            // throw std::runtime_error("");

            int split_index = min_cost_split_index;
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
            for (int object_index = 0; object_index < (int)geometry_array.size(); object_index++) {
                assigned_object_indices.push_back(object_index);
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

                std::cout << "node: " << node->_index << " object: " << object_id_bit << " "
                          << "max: " << node->_aabb_max.x << ", " << node->_aabb_max.y << ", " << node->_aabb_max.z << " ";
                std::cout << "min: " << node->_aabb_min.x << ", " << node->_aabb_min.y << ", " << node->_aabb_min.z << std::endl;

                std::cout << " index: ";
                std::cout << node->_index;
                std::cout << " left: ";
                if (node->_left) {
                    std::cout << node->_left->_index;
                }
                std::cout << " right: ";
                if (node->_right) {
                    std::cout << node->_right->_index;
                }
                std::cout << " hit: ";
                if (node->_hit) {
                    std::cout << node->_hit->_index;
                }
                std::cout << " miss: ";
                if (node->_miss) {
                    std::cout << node->_miss->_index;
                }
                std::cout << " object: " << object_id_bit;
                std::cout << " binary: " << binary_path;
                std::cout << std::endl;
            }
        }
    }
}
}