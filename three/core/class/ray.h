#pragma once
#include <glm/glm.hpp>

namespace three {
class Ray {
public:
    glm::vec3 _origin;
    glm::vec3 _direction;
    Ray(glm::vec3 origin, glm::vec3 direction);
    void set_origin(glm::vec3 origin);
    void set_direction(glm::vec3 direction);
    glm::vec3 point(float t);
};
}