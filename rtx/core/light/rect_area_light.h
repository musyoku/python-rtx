#pragma once
#include "../class/light.h"
#include "../header/glm.h"
#include <pybind11/pybind11.h>
#include <vector>

namespace rtx {
class RectAreaLight : public Light {
private:
    float _width;
    float _height;
    std::vector<glm::vec3i> _face_vertex_indices_array;
    std::vector<glm::vec4f> _vertex_array;

public:
    RectAreaLight(float width, float height, float brightness, pybind11::tuple color);
    RectAreaLight(float width, float height, float brightness, glm::vec3f color);
    int type() const override;
    int num_faces() const override;
    int num_vertices() const override;
    void serialize_vertices(rtx::array<RTXVertex>& array, int offset) const override;
    void serialize_faces(rtx::array<RTXFace>& array, int array_offset, int vertex_index_offset) const override;
    std::shared_ptr<Object> transoform(glm::mat4& transformation_matrix) const override;
};
}