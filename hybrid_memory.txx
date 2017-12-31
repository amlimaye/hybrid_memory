namespace utils {
    //ctor, allocates memory on the host
    template <typename T>
    hybrid_memory<T>::hybrid_memory(size_t a_size) {
        m_host_ptr = new T[a_size];
        m_device_ptr = nullptr;

        m_alloc_on_host = true;
        m_alloc_on_device = false;

        m_active_on = where_t::HOST;
        m_size = a_size;
    };

    //dtor, deallocates on the host and maybe the device
    template <typename T>
    hybrid_memory<T>::~hybrid_memory() {
        if (m_alloc_on_host) {
            delete[] m_host_ptr;
        }

        if (m_alloc_on_device) {
            cudaError_t errcode = cudaFree(m_device_ptr);
            if (errcode != cudaSuccess) {throw std::runtime_error("failed freeing device memory!");};
        }
    };

    //writes a default value to the whole block of memory if on the host
    template <typename T>
    void hybrid_memory<T>::fill(const T& elem) {
        if (m_active_on != where_t::HOST) {
            throw std::runtime_error("can only fill on the host!");
        }
        std::fill(m_host_ptr, m_host_ptr + m_size, elem);
    };
   
    //if active on the host, returns a host pointer 
    template <typename T>
    T* hybrid_memory<T>::host() const {
        if (m_active_on == where_t::HOST) {
            return m_host_ptr;
        } else {
            throw std::runtime_error("not active on the host!");
        }
    };

    //if active on the device, returns a device pointer    
    template <typename T>
    T* hybrid_memory<T>::device() const {
        if (m_active_on == where_t::DEVICE) {return m_device_ptr;}
        else {throw std::runtime_error("not active on the device!");}
    };
   
    //moves memory from host -> device 
    template <typename T>
    void hybrid_memory<T>::upload() {
        if (m_active_on != where_t::HOST) {throw std::runtime_error("not active on the host!");}

        //allocate device memory if it hasn't been done already
        if (!m_alloc_on_device) {
            cudaError_t errcode = cudaMalloc(&m_device_ptr, m_size * sizeof(T));
            if (errcode != cudaSuccess) {throw std::runtime_error("device memory allocation failed!");}
            m_alloc_on_device = true;
        }

        //copy over to device memory
        cudaError_t errcode = cudaMemcpy(m_device_ptr, m_host_ptr, m_size * sizeof(T), cudaMemcpyHostToDevice);
        if (errcode != cudaSuccess) {throw std::runtime_error("memcopy to device failed!");}

        m_active_on = where_t::DEVICE;
    };

    //moves memory from device -> host
    template <typename T>
    void hybrid_memory<T>::download() {
        if (m_active_on != where_t::DEVICE) {throw std::runtime_error("not active on the device!");}

        //copy over to host memory
        cudaError_t errcode = cudaMemcpy(m_host_ptr, m_device_ptr, m_size * sizeof(T), cudaMemcpyDeviceToHost);
        if (errcode != cudaSuccess) {throw std::runtime_error("memcopy to host failed!");}
        
        m_active_on = where_t::HOST;
    };
}
