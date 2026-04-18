#ifndef CB_RESULT_H
#define CB_RESULT_H

#include <string>
#include <vector>
#include <map>
#include <ctime>

namespace cb {

struct TestResult; // 前向声明

// 完整测试报告
struct BenchmarkReport {
    std::string version = "v0.1.0";
    std::time_t timestamp;
    std::string hostname;
    std::string platform;
    
    std::vector<TestResult> results;
    std::map<std::string, std::string> systemInfo;
};

// JSON序列化
std::string toJson(const BenchmarkReport& report);
std::string toJson(const TestResult& result);

// 保存到文件
void saveReport(const BenchmarkReport& report, const std::string& path);

// 加载历史数据
BenchmarkReport loadReport(const std::string& path);
std::vector<BenchmarkReport> loadHistory(const std::string& dir = "results");

// 比较两份报告
std::string compareReports(const BenchmarkReport& baseline, 
                          const BenchmarkReport& current);

} // namespace cb

#endif // CB_RESULT_H
