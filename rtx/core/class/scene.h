#pragma once
#include "object.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <vector>

namespace rtx {
class Scene {
private:
    bool _updated;

public:
    std::vector<std::shared_ptr<Object>> _object_array;
    std::vector<std::shared_ptr<ObjectGroup>> _object_group_array;
    rtxRGBAColor _ambient_color;

    Scene(pybind11::tuple ambient_color);
    void add(std::shared_ptr<Object> object);
    void add(std::shared_ptr<ObjectGroup> object);
    bool updated();
    void set_updated(bool updated);
    int num_triangles();
};
}