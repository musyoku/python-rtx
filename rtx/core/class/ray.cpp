#include "ray.h"

namespace rtx {
Ray::Ray(glm::vec3f origin, glm::vec3f direction)
{
    set_origin(origin);
    set_direction(direction);
}
void Ray::set_origin(glm::vec3f origin)
{
    _origin = origin;
}
void Ray::set_direction(glm::vec3f direction)
{
    _direction = direction;
}

glm::vec3f Ray::point(float t)
{
    return _origin + t * _direction;
}
}