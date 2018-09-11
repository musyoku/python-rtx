#pragma once
#include "object.h"
#include <memory>
#include <vector>

namespace rtx {
class Scene {
private:
    bool _updated;

public:
    std::vector<std::shared_ptr<Object>> _object_array;
    void add(std::shared_ptr<Object> object);
    bool updated();
    void set_updated(bool updated);
    int num_triangles();
};
}