#include "bvh.h"
#include "../../header/enum.h"
#include <algorithm>
#include <bitset>
#include <cassert>
#include <cfloat>
#include <iostream>
#include <utility>

namespace rtx {
using namespace bvh;
template <int L, typename T>
void merge_aabb_max(const glm::vec3f& a, const glm::vec<L, T>& b, glm::vec3f& dest)
{
    dest.x = a.x > b.x ? a.x : b.x;
    dest.y = a.y > b.y ? a.y : b.y;
    dest.z = a.z > b.z ? a.z : b.z;
}
template <int L, typename T>
void merge_aabb_min(const glm::vec3f& a, const glm::vec<L, T>& b, glm::vec3f& dest)
{
    dest.x = a.x < b.x ? a.x : b.x;
    dest.y = a.y < b.y ? a.y : b.y;
    dest.z = a.z < b.z ? a.z : b.z;
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
            return RTXAxisX;
        }
        return RTXAxisZ;
    }
    if (axis_length.y > axis_length.x) {
        if (axis_length.y > axis_length.z) {
            return RTXAxisY;
        }
        return RTXAxisZ;
    }
    if (axis_length.x > axis_length.z) {
        if (axis_length.y > axis_length.x) {
            return RTXAxisY;
        }
        return RTXAxisX;
    }
    return RTXAxisZ;
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
        merge_aabb_max(_aabb_max, va, _aabb_max);
        merge_aabb_min(_aabb_min, va, _aabb_min);

        auto& vb = geometry->_vertex_array[face[1]];
        merge_aabb_max(_aabb_max, vb, _aabb_max);
        merge_aabb_min(_aabb_min, vb, _aabb_min);

        auto& vc = geometry->_vertex_array[face[2]];
        merge_aabb_max(_aabb_max, vc, _aabb_max);
        merge_aabb_min(_aabb_min, vc, _aabb_min);
    }

    if ((int)assigned_face_indices.size() <= geometry->bvh_max_triangles_per_node()) {
        _is_leaf = true;
        _assigned_face_index_start = current_assigned_face_index_offset;
        _assigned_face_index_end = current_assigned_face_index_offset + assigned_face_indices.size() - 1;
        current_assigned_face_index_offset += (int)assigned_face_indices.size();
        return;
    }
    const glm::vec3f axis_length = _aabb_max - _aabb_min;
    int longest_axis = detect_longest_axis(axis_length);
    // if (_index == 0) {
    //     longest_axis = RTXAxisX;
    // }

    // std::cout << "longest: " << longest_axis << std::endl;
    std::vector<std::pair<int, float>> object_center_array;

    for (int face_index : assigned_face_indices) {
        auto& face = geometry->_face_vertex_indices_array.at(face_index);
        auto& va = geometry->_vertex_array[face[0]];
        auto& vb = geometry->_vertex_array[face[1]];
        auto& vc = geometry->_vertex_array[face[2]];
        auto center = (va + vb + vc) / 3.0f;
        if (longest_axis == RTXAxisX) {
            object_center_array.emplace_back(face_index, center.x);
        } else if (longest_axis == RTXAxisY) {
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

    glm::vec3f volume_a_max(0);
    glm::vec3f volume_a_min(0);
    glm::vec3f volume_b_max(0);
    glm::vec3f volume_b_min(0);
    // std::cout << "==============================================================" << std::endl;

    float min_cost = FLT_MAX;
    int min_cost_split_index = 0;
    if (true) {
        min_cost_split_index = object_center_array.size() / 2;
    } else {
        for (unsigned int split_index = 1; split_index <= object_center_array.size() - 1; split_index++) {
            int volume_a_num_faces = 0;
            int volume_b_num_faces = 0;
            for (unsigned int position = 0; position < split_index; position++) {
                int face_index = object_center_array[position].first;
                auto& face = geometry->_face_vertex_indices_array.at(face_index);

                glm::vec3f max = glm::vec3f(-FLT_MAX);
                glm::vec3f min = glm::vec3f(FLT_MAX);

                auto& va = geometry->_vertex_array[face[0]];
                merge_aabb_max(max, va, max);
                merge_aabb_min(min, va, min);

                auto& vb = geometry->_vertex_array[face[1]];
                merge_aabb_max(max, vb, max);
                merge_aabb_min(min, vb, min);

                auto& vc = geometry->_vertex_array[face[2]];
                merge_aabb_max(max, vc, max);
                merge_aabb_min(min, vc, min);

                if (position == 0) {
                    volume_a_max = max;
                    volume_a_min = min;
                } else {
                    merge_aabb_max(volume_a_max, max, volume_a_max);
                    merge_aabb_min(volume_a_min, min, volume_a_min);
                }
                volume_a_num_faces += 1;
            }
            for (unsigned int position = split_index; position < object_center_array.size(); position++) {
                int face_index = object_center_array[position].first;
                auto& face = geometry->_face_vertex_indices_array.at(face_index);

                glm::vec3f max = glm::vec3f(-FLT_MAX);
                glm::vec3f min = glm::vec3f(FLT_MAX);

                auto& va = geometry->_vertex_array[face[0]];
                merge_aabb_max(max, va, max);
                merge_aabb_min(min, va, min);

                auto& vb = geometry->_vertex_array[face[1]];
                merge_aabb_max(max, vb, max);
                merge_aabb_min(min, vb, min);

                auto& vc = geometry->_vertex_array[face[2]];
                merge_aabb_max(max, vc, max);
                merge_aabb_min(min, vc, min);

                if (position == split_index) {
                    volume_b_max = max;
                    volume_b_min = min;
                } else {
                    merge_aabb_max(volume_b_max, max, volume_b_max);
                    merge_aabb_min(volume_b_min, min, volume_b_min);
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
Node::Node(std::vector<int> assigned_face_indices,
    std::shared_ptr<SphereGeometry>& geometry)
{
    assert(assigned_face_indices.size() > 0);
    _assigned_face_indices = assigned_face_indices;
    _index = 0;
    _assigned_face_index_start = 0;
    _assigned_face_index_end = 0;

    // AABBは使わない
    _aabb_max = glm::vec3f(0.0f);
    _aabb_min = glm::vec3f(0.0f);

    _is_leaf = true;
}
Node::Node(std::vector<int> assigned_face_indices,
    std::shared_ptr<CylinderGeometry>& geometry)
{
    assert(assigned_face_indices.size() > 0);
    _assigned_face_indices = assigned_face_indices;
    _index = 0;
    _assigned_face_index_start = 0;
    _assigned_face_index_end = 0;

    // AABBは使わない
    _aabb_max = glm::vec3f(0.0f);
    _aabb_min = glm::vec3f(0.0f);

    _is_leaf = true;
}
Node::Node(std::vector<int> assigned_face_indices,
    std::shared_ptr<ConeGeometry>& geometry)
{
    assert(assigned_face_indices.size() > 0);
    _assigned_face_indices = assigned_face_indices;
    _index = 0;
    _assigned_face_index_start = 0;
    _assigned_face_index_end = 0;

    // AABBは使わない
    _aabb_max = glm::vec3f(0.0f);
    _aabb_min = glm::vec3f(0.0f);

    _is_leaf = true;
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
BVH::BVH(std::shared_ptr<Geometry>& geometry)
{
    _geometry = geometry;
    std::vector<int> assigned_face_indices;
    if (geometry->type() == RTXGeometryTypeStandard) {
        std::shared_ptr<StandardGeometry> standard = std::static_pointer_cast<StandardGeometry>(geometry);
        for (int face_index = 0; face_index < (int)standard->_face_vertex_indices_array.size(); face_index++) {
            assigned_face_indices.push_back(face_index);
        }
        _current_node_index = 0;
        _current_assigned_face_index_offset = 0;
        _root = std::make_shared<Node>(assigned_face_indices, standard, _current_node_index, _current_assigned_face_index_offset);
        _root->set_hit_and_miss_links();

        _num_nodes = _root->num_children() + 1;
        return;
    }
    if (geometry->type() == RTXGeometryTypeSphere) {
        std::shared_ptr<SphereGeometry> sphere = std::static_pointer_cast<SphereGeometry>(geometry);
        std::vector<int> assigned_face_indices;
        for (int face_index = 0; face_index < geometry->num_faces(); face_index++) {
            assigned_face_indices.push_back(face_index);
        }
        _current_node_index = 0;
        _current_assigned_face_index_offset = 0;
        _root = std::make_shared<Node>(assigned_face_indices, sphere);
        _root->set_hit_and_miss_links();

        _num_nodes = _root->num_children() + 1;
        return;
    }
    if (geometry->type() == RTXGeometryTypeCylinder) {
        std::shared_ptr<CylinderGeometry> cylinder = std::static_pointer_cast<CylinderGeometry>(geometry);
        std::vector<int> assigned_face_indices;
        for (int face_index = 0; face_index < geometry->num_faces(); face_index++) {
            assigned_face_indices.push_back(face_index);
        }
        _current_node_index = 0;
        _current_assigned_face_index_offset = 0;
        _root = std::make_shared<Node>(assigned_face_indices, cylinder);
        _root->set_hit_and_miss_links();

        _num_nodes = _root->num_children() + 1;
        return;
    }
    if (geometry->type() == RTXGeometryTypeCone) {
        std::shared_ptr<ConeGeometry> cone = std::static_pointer_cast<ConeGeometry>(geometry);
        std::vector<int> assigned_face_indices;
        for (int face_index = 0; face_index < geometry->num_faces(); face_index++) {
            assigned_face_indices.push_back(face_index);
        }
        _current_node_index = 0;
        _current_assigned_face_index_offset = 0;
        _root = std::make_shared<Node>(assigned_face_indices, cone);
        _root->set_hit_and_miss_links();

        _num_nodes = _root->num_children() + 1;
        return;
    }
}
int BVH::num_nodes()
{
    return _num_nodes;
}
void BVH::serialize_nodes(rtx::array<rtxThreadedBVHNode>& node_array, int serialization_offset)
{
    std::vector<std::shared_ptr<Node>> children = { _root };
    _root->collect_children(children);
    for (auto& node : children) {
        int pos = node->_index + serialization_offset;

        rtxThreadedBVHNode cuda_node;
        cuda_node.hit_node_index = node->_hit ? node->_hit->_index : -1;
        cuda_node.miss_node_index = node->_miss ? node->_miss->_index : -1;
        cuda_node.assigned_face_index_start = node->_assigned_face_index_start;
        cuda_node.assigned_face_index_end = node->_assigned_face_index_end;
        cuda_node.aabb_max.x = node->_aabb_max.x;
        cuda_node.aabb_max.y = node->_aabb_max.y;
        cuda_node.aabb_max.z = node->_aabb_max.z;
        cuda_node.aabb_min.x = node->_aabb_min.x;
        cuda_node.aabb_min.y = node->_aabb_min.y;
        cuda_node.aabb_min.z = node->_aabb_min.z;

        node_array[pos] = cuda_node;

        // printf("node: %d face_start: %d face_end: %d max: (%f, %f, %f) min: (%f, %f, %f)\n", node->_index, node->_assigned_face_index_start, node->_assigned_face_index_end, node->_aabb_max.x, node->_aabb_max.y, node->_aabb_max.z, node->_aabb_min.x, node->_aabb_min.y, node->_aabb_min.z);
        // printf("    hit: %d miss: %d left: %d right: %d\n", (node->_hit ? node->_hit->_index : -1), (node->_miss ? node->_miss->_index : -1), (node->_left ? node->_left->_index : -1), (node->_right ? node->_right->_index : -1));
    }
}
void BVH::serialize_faces(rtx::array<rtxFaceVertexIndex>& buffer, int serialization_offset)
{
    std::vector<std::shared_ptr<bvh::Node>> leaves;
    collect_leaves(leaves);
    assert(_geometry.expired() == false);

    auto geometry = _geometry.lock();
    assert(geometry);

    if (geometry->type() == RTXGeometryTypeStandard) {
        std::shared_ptr<StandardGeometry> standard = std::static_pointer_cast<StandardGeometry>(geometry);
        auto& face_vertex_indices_array = standard->_face_vertex_indices_array;
        // printf("============================================================\n");
        // for (int face_index = 0; face_index < face_vertex_indices_array.size(); face_index++) {
        //     auto& face = face_vertex_indices_array[face_index];
        //     // printf("[%d] (%d, %d, %d)\n", face_index, face[0], face[1], face[2]);
        // }
        for (auto& node : leaves) {
            int pos = node->_assigned_face_index_start + serialization_offset;
            for (int face_index : node->_assigned_face_indices) {
                glm::vec3i face = face_vertex_indices_array[face_index];
                buffer[pos] = { face[0], face[1], face[2], -1 };
                pos++;
                // printf("[%d] (%d, %d, %d)\n", face_index, face[0], face[1], face[2]);
            }
        }
        return;
    }
    if (geometry->type() == RTXGeometryTypeSphere) {
        int pos = serialization_offset;
        // vertex 0: center
        // vertex 1: radius
        buffer[pos] = { 0, 1, -1, -1 };
        return;
    }
    if (geometry->type() == RTXGeometryTypeCylinder) {
        int pos = serialization_offset;
        // vertex 0: cylinder parameter
        buffer[pos + 0] = { 0, -1, -1, -1 };
        // Set transformation matrix
        buffer[pos + 1] = { 1, 2, 3, -1 };
        // Set inverse transformation matrix
        buffer[pos + 2] = { 4, 5, 6, -1 };
        return;
    }
    if (geometry->type() == RTXGeometryTypeCone) {
        int pos = serialization_offset;
        // vertex 0: cylinder parameter
        buffer[pos + 0] = { 0, -1, -1, -1 };
        // Set transformation matrix
        buffer[pos + 1] = { 1, 2, 3, -1 };
        // Set inverse transformation matrix
        buffer[pos + 2] = { 4, 5, 6, -1 };
        return;
    }
}
void BVH::collect_leaves(std::vector<std::shared_ptr<bvh::Node>>& leaves)
{
    if (_root->_is_leaf) {
        leaves.push_back(_root);
        return;
    }
    _root->collect_leaves(leaves);
}
}