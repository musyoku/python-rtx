namespace rtx {

inline float pow2(float x)
{
    return x * x;
}
inline float pow3(float x)
{
    return x * x * x;
}
inline float pow4(float x)
{
    return x * x * x * x;
}
inline float pow5(float x)
{
    return x * x * x * x * x;
}
inline float clamp(float x, float a, float b)
{
    return x < a ? a : x > b ? b : x;
}
inline float saturate(float x)
{
    return x < 0.f ? 0.f : x > 1.f ? 1.f : x;
}
inline float recip(float x)
{
    return 1.f / x;
}
inline float mix(float a, float b, float t)
{
    return a * (1.f - t) + b * t;
}
inline float step(float edge, float x)
{
    return (x < edge) ? 0.f : 1.f;
}
inline float smoothstep(float a, float b, float t)
{
    if (a >= b)
        return 0.f;
    float x = saturate((t - a) / (b - a));
    return x * x * (3.f - 2.f * t);
}
inline float radians(float deg)
{
    return (deg / 180.f) * M_PI;
}
inline float degrees(float rad)
{
    return (rad / M_PI) * 180.f;
}
}