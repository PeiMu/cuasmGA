#include <cuda_runtime.h>
#include <cupti.h>
#include <vector>
#include <string>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define CUPTI_CALL(call)                                                \
do {                                                                    \
    CUptiResult _status = call;                                         \
    if (_status != CUPTI_SUCCESS) {                                     \
        const char *errstr;                                             \
        cuptiGetResultString(_status, &errstr);                         \
        throw std::runtime_error(std::string("CUPTI error: ") + errstr); \
    }                                                                   \
} while (0)

struct KernelMetrics {
    std::string name;
    double duration_us;
    uint64_t start;
    uint64_t end;
    uint32_t gridX;
    uint32_t gridY;
    uint32_t gridZ;
    uint32_t blockX;
    uint32_t blockY;
    uint32_t blockZ;
};

class CuptiProfiler {
private:
    std::vector<KernelMetrics> metrics;
    bool is_profiling;
    
    static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
        *size = 1024 * 1024;  // 1MB buffer
        *buffer = (uint8_t *)malloc(*size);
        *maxNumRecords = 0;
    }
    
    static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, 
                                         uint8_t *buffer, size_t size, size_t validSize) {
        CUpti_Activity *record = NULL;
        CUptiResult status;
        
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
                    CUpti_ActivityKernel4 *kernel = (CUpti_ActivityKernel4 *)record;
                    
                    KernelMetrics metric;
                    metric.name = std::string(kernel->name);
                    metric.start = kernel->start;
                    metric.end = kernel->end;
                    metric.duration_us = (kernel->end - kernel->start) / 1000.0;
                    metric.gridX = kernel->gridX;
                    metric.gridY = kernel->gridY;
                    metric.gridZ = kernel->gridZ;
                    metric.blockX = kernel->blockX;
                    metric.blockY = kernel->blockY;
                    metric.blockZ = kernel->blockZ;
                    
                    getInstance()->metrics.push_back(metric);
                }
            } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            }
        } while (status == CUPTI_SUCCESS);
        
        free(buffer);
    }
    
    static CuptiProfiler* instance;
    
    CuptiProfiler() : is_profiling(false) {}
    
public:
    static CuptiProfiler* getInstance() {
        if (instance == nullptr) {
            instance = new CuptiProfiler();
        }
        return instance;
    }
    
    void start() {
        if (is_profiling) {
            throw std::runtime_error("Profiler is already running");
        }
        
        metrics.clear();
        CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
        is_profiling = true;
    }
    
    void stop() {
        if (!is_profiling) {
            throw std::runtime_error("Profiler is not running");
        }
        
        cudaDeviceSynchronize();
        CUPTI_CALL(cuptiActivityFlushAll(0));
        cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
        is_profiling = false;
    }
    
    py::dict get_metrics() {
        py::dict result;
        py::list kernel_list;
        
        for (const auto& metric : metrics) {
            py::dict kernel_dict;
            kernel_dict["name"] = metric.name;
            kernel_dict["duration_us"] = metric.duration_us;
            kernel_dict["grid"] = py::make_tuple(metric.gridX, metric.gridY, metric.gridZ);
            kernel_dict["block"] = py::make_tuple(metric.blockX, metric.blockY, metric.blockZ);
            kernel_list.append(kernel_dict);
        }
        
        result["kernels"] = kernel_list;
        result["count"] = metrics.size();
        
        if (metrics.size() > 0) {
            double total_time = 0;
            for (const auto& m : metrics) {
                total_time += m.duration_us;
            }
            result["total_time_us"] = total_time;
            result["avg_time_us"] = total_time / metrics.size();
        }
        
        return result;
    }
    
    void clear() {
        metrics.clear();
    }
};

CuptiProfiler* CuptiProfiler::instance = nullptr;

PYBIND11_MODULE(cupti_profiler, m) {
    py::class_<CuptiProfiler>(m, "CuptiProfiler")
        .def_static("get_instance", &CuptiProfiler::getInstance, py::return_value_policy::reference)
        .def("start", &CuptiProfiler::start)
        .def("stop", &CuptiProfiler::stop)
        .def("get_metrics", &CuptiProfiler::get_metrics)
        .def("clear", &CuptiProfiler::clear);
}
