#include "geometry.h"

namespace rtx {
int Geometry::num_bvh_split()
{
    return _num_bvh_split;
}
std::shared_ptr<bvh::geometry::GeometryBVH> Geometry::bvh()
{
    return _bvh;
}
}