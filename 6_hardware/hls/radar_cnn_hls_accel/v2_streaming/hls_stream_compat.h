#ifndef RADAR_CNN_HLS_STREAM_COMPAT_H
#define RADAR_CNN_HLS_STREAM_COMPAT_H

// Host-only builds (-DRADAR_CNN_SOFTWARE_FLOAT) run without a Vitis
// installation, so hls::stream is replaced by an unbounded std::queue.
// FIFO depth / back-pressure is NOT modeled on the host; it only exists
// in HLS csim, cosim and hardware.
#ifdef RADAR_CNN_SOFTWARE_FLOAT
#include <queue>
namespace hls {
template<typename T>
class stream {
public:
    void write(const T &value) { values.push(value); }
    T read() { T value = values.front(); values.pop(); return value; }
    bool empty() const { return values.empty(); }
private:
    std::queue<T> values;
};
}
#else
#include <hls_stream.h>
#endif

#endif
