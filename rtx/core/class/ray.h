#pragma once
#include "../header/glm.h"

namespace rtx {
class Ray {
public:
    glm::vec3f _origin;
    glm::vec3f _direction;
    Ray(glm::vec3f origin, glm::vec3f direction);
    void set_origin(glm::vec3f origin);
    void set_direction(glm::vec3f direction);
    glm::vec3f point(float t);
};
}