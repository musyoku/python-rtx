#include "ray.h"

namespace three {
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
}