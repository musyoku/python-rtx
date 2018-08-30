#include "geometry.h"
#include <cassert>

namespace rtx {
int Geometry::num_bvh_split()
{
    return _num_bvh_split;
}
bvh::geometry::GeometryBVH* Geometry::bvh()
{
    assert(!!_bvh == true);
    return _bvh.get();
}
}