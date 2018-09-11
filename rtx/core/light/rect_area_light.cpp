#include "rect_area_light.h"
#include "../header/enum.h"

namespace rtx {
RectAreaLight::RectAreaLight(float width, float height, float brightness, pybind11::tuple color)
{
    _brightness = brightness;

    _face_vertex_indices_array.emplace_back(glm::vec3i(0, 1, 2));
    _face_vertex_indices_array.emplace_back(glm::vec3i(2, 1, 3));

    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, -height / 2.0f, 0.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, -height / 2.0f, 0.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, height / 2.0f, 0.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, height / 2.0f, 0.0f, 1.0f));

    set_color(color);
}
RectAreaLight::RectAreaLight(float width, float height, float brightness, glm::vec3f color)
{
    _brightness = brightness;

    _face_vertex_indices_array.emplace_back(glm::vec3i(0, 1, 2));
    _face_vertex_indices_array.emplace_back(glm::vec3i(2, 1, 3));

    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, -height / 2.0f, 0.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, -height / 2.0f, 0.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(-width / 2.0f, height / 2.0f, 0.0f, 1.0f));
    _vertex_array.emplace_back(glm::vec4f(width / 2.0f, height / 2.0f, 0.0f, 1.0f));

    _color = color;
}
int RectAreaLight::type() const
{
    return RTXObjectTypeRectAreaLight;
}
int RectAreaLight::num_faces() const
{
    return _face_vertex_indices_array.size();
}
int RectAreaLight::num_vertices() const
{
    return _vertex_array.size();
}
void RectAreaLight::serialize_vertices(rtx::array<RTXVertex>& buffer, int array_offset) const
{
    for (int j = 0; j < _vertex_array.size(); j++) {
        auto& vertex = _vertex_array[j];
        buffer[j + array_offset] = { vertex.x, vertex.y, vertex.z };
    }
}
void RectAreaLight::serialize_faces(rtx::array<RTXFace>& buffer, int array_offset, int vertex_index_offset) const
{
    for (int j = 0; j < _face_vertex_indices_array.size(); j++) {
        auto& face = _face_vertex_indices_array[j];
        buffer[j + array_offset] = { face[0] + vertex_index_offset, face[1] + vertex_index_offset, face[2] + vertex_index_offset };
    }
}
std::shared_ptr<Object> RectAreaLight::transoform(glm::mat4& transformation_matrix) const
{
    auto light = std::make_shared<RectAreaLight>(_brightness, _width, _height, _color);
    light->_face_vertex_indices_array = _face_vertex_indices_array;
    light->_vertex_array.resize(_vertex_array.size());
    for (int index = 0; index < _vertex_array.size(); index++) {
        auto& vertex = _vertex_array[index];
        glm::vec4f v = transformation_matrix * vertex;
        light->_vertex_array[index] = v;
    }
    return light;
}
}