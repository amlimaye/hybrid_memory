#include <stddef.h>
#include <stdexcept>
#include <algorithm>

namespace utils {

enum struct where_t {HOST, DEVICE};

template <typename T>
class hybrid_memory {
public:
    hybrid_memory(size_t a_size);
    ~hybrid_memory();

    T*          host() const;
    T*          device() const;

    void        upload();
    void        download();

    void        fill(const T& elem);
    size_t      size() const {return m_size;};

private:
    T*              m_host_ptr;
    T*              m_device_ptr;
    where_t         m_active_on;
    const size_t    m_size;
    bool            m_alloc_on_host;
    bool            m_alloc_on_device;
};

}

#include "hybrid_memory.txx"
