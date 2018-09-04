#include "geometry.h"
#include <cassert>

namespace rtx {
bool Geometry::bvh_enabled() const
{
    return false;
}
int Geometry::bvh_max_triangles_per_node() const
{
    return -1;
}
}