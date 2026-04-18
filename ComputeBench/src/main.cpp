#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unistd.h>
#include <sys/utsname.h>

#include "core/benchmark.h"
#include "core/result.h"

using namespace cb;

// 获取系统信息
void collectSystemInfo(BenchmarkReport& report) {
    struct utsname uts;
    if (uname(&uts) == 0) {
        report.platform = std::string(uts.sysname) + " " + uts.machine;
    }
    
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        report.hostname = hostname;
    }
    
    // 读取CPU信息
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo) {
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos) {
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    report.systemInfo["cpu_model"] = line.substr(pos + 2);
                }
                break;
            }
        }
    }
    
    // 读取内存信息
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal") != std::string::npos) {
                size_t pos = line.find(":");
                if (pos != std::string::npos) {
                    report.systemInfo["memory"] = line.substr(pos + 1);
                }
                break;
            }
        }
    }
}

// 生成Markdown报告
void generateMarkdownReport(const BenchmarkReport& report) {
    std::ofstream md("reports/latest.md");
    if (!md) return;
    
    md << "# ComputeBench Report\n\n";
    md << "**Version**: " << report.version << "  \n";
    md << "**Time**: " << std::ctime(&report.timestamp);
    md << "**Platform**: " << report.platform << "\n\n";
    
    md << "## System Information\n\n";
    for (const auto& [key, value] : report.systemInfo) {
        md << "- **" << key << "**: " << value << "\n";
    }
    md << "\n";
    
    md << "## Results\n\n";
    md << "| Test | Value | Unit | Min | Max | StdDev |\n";
    md << "|------|-------|------|-----|-----|--------|\n";
    
    for (const auto& r : report.results) {
        md << "| " << r.name << " | ";
        md << std::fixed << std::setprecision(2) << r.value << " | ";
        md << r.unit << " | ";
        md << r.min << " | ";
        md << r.max << " | ";
        md << r.stddev << " |\n";
    }
    md << "\n";
}

int main(int argc, char* argv[]) {
    std::cout << "ComputeBench v0.1.0 - High Performance Computing Benchmark\n";
    std::cout << "=========================================================\n\n";
    
    // 检查参数
    std::string specific_test;
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg.find("--test=") == 0) {
            specific_test = arg.substr(7);
        }
    }
    
    // 准备报告
    BenchmarkReport report;
    report.timestamp = std::time(nullptr);
    collectSystemInfo(report);
    
    // 运行测试
    auto& registry = BenchmarkRegistry::instance();
    auto benchmarks = registry.getAll();
    
    if (specific_test.empty()) {
        std::cout << "Running all " << benchmarks.size() << " benchmarks...\n\n";
        for (auto* bench : benchmarks) {
            if (bench->isAvailable()) {
                std::cout << "Running: " << bench->name() << "...\n";
                auto result = bench->run();
                report.results.push_back(result);
                std::cout << "  Result: " << std::fixed << std::setprecision(2) 
                         << result.value << " " << result.unit << "\n\n";
            }
        }
    } else {
        auto* bench = registry.getByName(specific_test);
        if (bench && bench->isAvailable()) {
            std::cout << "Running: " << bench->name() << "...\n";
            auto result = bench->run();
            report.results.push_back(result);
        } else {
            std::cerr << "Test not found or not available: " << specific_test << "\n";
            return 1;
        }
    }
    
    // 保存结果
    std::string json_path = "results/latest.json";
    saveReport(report, json_path);
    std::cout << "Results saved to: " << json_path << "\n";
    
    // 生成Markdown报告
    generateMarkdownReport(report);
    std::cout << "Report saved to: reports/latest.md\n";
    
    return 0;
}
