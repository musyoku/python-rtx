#pragma once
#include "object.h"
#include <memory>
#include <pybind11/pybind11.h>

namespace rtx {
class Light : public Object {
protected:
    void update_model_matrix();
    glm::vec3f _position;
    glm::vec3f _rotation_rad;
    glm::vec3f _color;
    float _brightness;

public:
    glm::mat4 _model_matrix;
    bool is_light() const override;
    float brightness() const;
    void set_brightness(float brightness);
    void set_color(pybind11::tuple color);
    void set_position(pybind11::tuple position);
    void set_rotation(pybind11::tuple rotation_rad);
};
}