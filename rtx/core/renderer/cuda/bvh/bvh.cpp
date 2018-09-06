#include "bvh.h"
#include <algorithm>
#include <bitset>
#include <cassert>
#include <cfloat>
#include <iostream>
#include <utility>

namespace rtx {
using namespace bvh;
template <int L, typename T>
glm::vec3f merge_aabb_max(const glm::vec3f& a, const glm::vec<L, T>& b)
{
    glm::vec3f max;
    max.x = a.x > b.x ? a.x : b.x;
    max.y = a.y > b.y ? a.y : b.y;
    max.z = a.z > b.z ? a.z : b.z;
    return max;
}
template <int L, typename T>
glm::vec3f merge_aabb_min(const glm::vec3f& a, const glm::vec<L, T>& b)
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
Node::Node(std::vector<int> assigned_face_indices,
    std::shared_ptr<StandardGeometry>& geometry,
    int& current_node_index,
    int& current_assigned_face_index_offset)
{
    // std::cout << "===================================" << std::endl;
    assert(assigned_face_indices.size() > 0);
    _assigned_face_indices = assigned_face_indices;
    _index = current_node_index;
    _assigned_face_index_start = -1;
    _assigned_face_index_end = -1;
    _is_leaf = false;
    // std::cout << "id: " << _index << std::endl;
    current_node_index++;
    _aabb_max = glm::vec3f(-FLT_MAX);
    _aabb_min = glm::vec3f(FLT_MAX);

    for (int face_index : assigned_face_indices) {
        auto& face = geometry->_face_vertex_indices_array.at(face_index);
        auto& va = geometry->_vertex_array[face[0]];
        _aabb_max = merge_aabb_max(_aabb_max, va);
        _aabb_min = merge_aabb_min(_aabb_min, va);

        auto& vb = geometry->_vertex_array[face[1]];
        _aabb_max = merge_aabb_max(_aabb_max, vb);
        _aabb_min = merge_aabb_min(_aabb_min, vb);

        auto& vc = geometry->_vertex_array[face[2]];
        _aabb_max = merge_aabb_max(_aabb_max, vc);
        _aabb_min = merge_aabb_min(_aabb_min, vc);
    }

    if (assigned_face_indices.size() <= geometry->bvh_max_triangles_per_node()) {
        _is_leaf = true;
        _assigned_face_index_start = current_assigned_face_index_offset;
        _assigned_face_index_end = current_assigned_face_index_offset + assigned_face_indices.size() - 1;
        current_assigned_face_index_offset += assigned_face_indices.size();
        return;
    }
    const glm::vec3f axis_length = _aabb_max - _aabb_min;
    int longest_axis = detect_longest_axis(axis_length);
    // if (_index == 0) {
    //     longest_axis = RTX_AXIS_X;
    // }

    // std::cout << "longest: " << longest_axis << std::endl;
    std::vector<std::pair<int, float>> object_center_array;

    for (int face_index : assigned_face_indices) {
        auto& face = geometry->_face_vertex_indices_array.at(face_index);
        auto& va = geometry->_vertex_array[face[0]];
        auto& vb = geometry->_vertex_array[face[1]];
        auto& vc = geometry->_vertex_array[face[2]];
        auto center = (va + vb + vc) / 3.0f;
        if (longest_axis == RTX_AXIS_X) {
            object_center_array.emplace_back(face_index, center.x);
        } else if (longest_axis == RTX_AXIS_Y) {
            object_center_array.emplace_back(face_index, center.y);
        } else {
            object_center_array.emplace_back(face_index, center.z);
        }
    }

    std::sort(object_center_array.begin(), object_center_array.end(), compare_position);
    // std::cout << "sort:" << std::endl;
    // for (auto& pair : object_center_array) {
    //     std::cout << pair.first << ": " << pair.second << std::endl;
    // }
    float whole_surface_area = compute_surface_area(_aabb_max, _aabb_min);
    // std::cout << "whole_surface_area: " << whole_surface_area << std::endl;

    glm::vec3f volume_a_max(FLT_MAX);
    glm::vec3f volume_a_min(FLT_MAX);
    glm::vec3f volume_b_max(FLT_MAX);
    glm::vec3f volume_b_min(FLT_MAX);
    // std::cout << "==============================================================" << std::endl;

    float min_cost = FLT_MAX;
    int min_cost_split_index = 0;
    if (true) {
        min_cost_split_index = object_center_array.size() / 2;
    } else {
        for (int split_index = 1; split_index <= object_center_array.size() - 1; split_index++) {
            int volume_a_num_faces = 0;
            int volume_b_num_faces = 0;
            for (int position = 0; position < split_index; position++) {
                int face_index = object_center_array[position].first;
                auto& face = geometry->_face_vertex_indices_array.at(face_index);

                glm::vec3f max = glm::vec3f(-FLT_MAX);
                glm::vec3f min = glm::vec3f(FLT_MAX);

                auto& va = geometry->_vertex_array[face[0]];
                max = merge_aabb_max(max, va);
                min = merge_aabb_min(min, va);

                auto& vb = geometry->_vertex_array[face[1]];
                max = merge_aabb_max(max, vb);
                min = merge_aabb_min(min, vb);

                auto& vc = geometry->_vertex_array[face[2]];
                max = merge_aabb_max(max, vc);
                min = merge_aabb_min(min, vc);

                if (position == 0) {
                    volume_a_max = max;
                    volume_a_min = min;
                } else {
                    volume_a_max = merge_aabb_max(volume_a_max, max);
                    volume_a_min = merge_aabb_min(volume_a_min, min);
                }
                volume_a_num_faces += 1;
            }
            for (int position = split_index; position < object_center_array.size(); position++) {
                int face_index = object_center_array[position].first;
                auto& face = geometry->_face_vertex_indices_array.at(face_index);

                glm::vec3f max = glm::vec3f(-FLT_MAX);
                glm::vec3f min = glm::vec3f(FLT_MAX);

                auto& va = geometry->_vertex_array[face[0]];
                max = merge_aabb_max(max, va);
                min = merge_aabb_min(min, va);

                auto& vb = geometry->_vertex_array[face[1]];
                max = merge_aabb_max(max, vb);
                min = merge_aabb_min(min, vb);

                auto& vc = geometry->_vertex_array[face[2]];
                max = merge_aabb_max(max, vc);
                min = merge_aabb_min(min, vc);

                if (position == split_index) {
                    volume_b_max = max;
                    volume_b_min = min;
                } else {
                    volume_b_max = merge_aabb_max(volume_b_max, max);
                    volume_b_min = merge_aabb_min(volume_b_min, min);
                }
                volume_b_num_faces += 1;
            }
            float surface_a = compute_surface_area(volume_a_max, volume_a_min);
            float surface_b = compute_surface_area(volume_b_max, volume_b_min);
            float cost = surface_a * volume_a_num_faces + surface_b * volume_b_num_faces;
            if (cost < min_cost) {
                min_cost = cost;
                min_cost_split_index = split_index;
            }
        }
    }

    int split_index = min_cost_split_index;
    // std::cout << "split: " << split_index << std::endl;
    std::vector<int> left_assigned_indices;
    std::vector<int> right_assigned_indices;
    for (int n = 0; n < split_index; n++) {
        auto& pair = object_center_array.at(n);
        left_assigned_indices.push_back(pair.first);
    }
    for (int n = split_index; n < (int)assigned_face_indices.size(); n++) {
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
    _left = std::make_shared<Node>(left_assigned_indices, geometry, current_node_index, current_assigned_face_index_offset);
    _right = std::make_shared<Node>(right_assigned_indices, geometry, current_node_index, current_assigned_face_index_offset);
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
void Node::collect_leaves(std::vector<std::shared_ptr<Node>>& leaves)
{
    if (_left) {
        if (_left->_is_leaf) {
            leaves.push_back(_left);
        } else {
            _left->collect_leaves(leaves);
        }
    }
    if (_right) {
        if (_right->_is_leaf) {
            leaves.push_back(_right);
        } else {
            _right->collect_leaves(leaves);
        }
    }
}
BVH::BVH(std::shared_ptr<StandardGeometry>& geometry)
{
    std::vector<int> assigned_face_indices;
    for (int face_index = 0; face_index < (int)geometry->_face_vertex_indices_array.size(); face_index++) {
        assigned_face_indices.push_back(face_index);
    }
    _current_node_index = 0;
    _current_assigned_face_index_offset = 0;
    _root = std::make_shared<Node>(assigned_face_indices, geometry, _current_node_index, _current_assigned_face_index_offset);
    _root->set_hit_and_miss_links();

    _num_nodes = _root->num_children() + 1;
}
int BVH::num_nodes()
{
    return _num_nodes;
}
void BVH::serialize(rtx::array<RTXThreadedBVHNode>& node_array, int offset)
{
    std::vector<std::shared_ptr<Node>> children = { _root };
    _root->collect_children(children);
    for (auto& node_obj : children) {
        int j = node_obj->_index + offset;

        RTXThreadedBVHNode node;
        node.hit_node_index = node_obj->_hit ? node_obj->_hit->_index : -1;
        node.miss_node_index = node_obj->_miss ? node_obj->_miss->_index : -1;
        node.assigned_face_index_start = node_obj->_assigned_face_index_start;
        node.assigned_face_index_end = node_obj->_assigned_face_index_end;
        node.aabb_max.x = node_obj->_aabb_max.x;
        node.aabb_max.y = node_obj->_aabb_max.y;
        node.aabb_max.z = node_obj->_aabb_max.z;
        node.aabb_min.x = node_obj->_aabb_min.x;
        node.aabb_min.y = node_obj->_aabb_min.y;
        node.aabb_min.z = node_obj->_aabb_min.z;

        node_array[j] = node;

        // printf("node_obj: %d face_start: %d face_end: %d max: (%f, %f, %f) min: (%f, %f, %f)\n", node_obj->_index, node_obj->_assigned_face_index_start, node_obj->_assigned_face_index_end, node_obj->_aabb_max.x, node_obj->_aabb_max.y, node_obj->_aabb_max.z, node_obj->_aabb_min.x, node_obj->_aabb_min.y, node_obj->_aabb_min.z);
        // printf("    hit: %d miss: %d left: %d right: %d\n", (node_obj->_hit ? node_obj->_hit->_index : -1), (node_obj->_miss ? node_obj->_miss->_index : -1), (node_obj->_left ? node_obj->_left->_index : -1), (node_obj->_right ? node_obj->_right->_index : -1));
    }
}
void BVH::collect_leaves(std::vector<std::shared_ptr<bvh::Node>>& leaves)
{
    _root->collect_leaves(leaves);
}
}