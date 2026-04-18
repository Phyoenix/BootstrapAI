#include "result.h"
#include "benchmark.h"
#include <sstream>
#include <iomanip>
#include <fstream>
#include <iostream>

namespace cb {

std::string toJson(const TestResult& r) {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"name\": \"" << r.name << "\",\n";
    oss << "  \"unit\": \"" << r.unit << "\",\n";
    oss << "  \"value\": " << std::fixed << std::setprecision(2) << r.value << ",\n";
    oss << "  \"min\": " << r.min << ",\n";
    oss << "  \"max\": " << r.max << ",\n";
    oss << "  \"stddev\": " << r.stddev << ",\n";
    oss << "  \"samples\": " << r.samples << "\n";
    oss << "}";
    return oss.str();
}

std::string toJson(const BenchmarkReport& report) {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"version\": \"" << report.version << "\",\n";
    oss << "  \"timestamp\": " << report.timestamp << ",\n";
    oss << "  \"hostname\": \"" << report.hostname << "\",\n";
    oss << "  \"platform\": \"" << report.platform << "\",\n";
    oss << "  \"systemInfo\": {\n";
    for (auto it = report.systemInfo.begin(); it != report.systemInfo.end(); ++it) {
        oss << "    \"" << it->first << "\": \"" << it->second << "\"";
        if (std::next(it) != report.systemInfo.end()) oss << ",";
        oss << "\n";
    }
    oss << "  },\n";
    oss << "  \"results\": [\n";
    for (size_t i = 0; i < report.results.size(); ++i) {
        oss << "    " << toJson(report.results[i]);
        if (i < report.results.size() - 1) oss << ",";
        oss << "\n";
    }
    oss << "  ]\n";
    oss << "}";
    return oss.str();
}

void saveReport(const BenchmarkReport& report, const std::string& path) {
    std::ofstream ofs(path);
    if (ofs) {
        ofs << toJson(report);
    }
}

} // namespace cb
