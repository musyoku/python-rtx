#include "ray.h"

namespace rtx {
Ray::Ray(glm::vec3 origin, glm::vec3 direction)
{
    set_origin(origin);
    set_direction(direction);
}
void Ray::set_origin(glm::vec3 origin)
{
    _origin = origin;
}
void Ray::set_direction(glm::vec3 direction)
{
    _direction = direction;
}

glm::vec3 Ray::point(float t)
{
    return _origin + t * _direction;
}
}