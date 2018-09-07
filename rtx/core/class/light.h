#pragma once

namespace rtx {
class Light {
protected:
    float _brightness;

public:
    virtual float brightness() const;
    virtual int type() const = 0;
};
}