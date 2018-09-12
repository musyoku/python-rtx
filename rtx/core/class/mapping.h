#pragma once

namespace rtx {
class Mapping {
    virtual int type() const = 0;
    virtual int texture_index() const;
};
}