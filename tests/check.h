// Minimal result accounting for the test runner.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace mcc::test {

struct Check {
    std::string name;
    uint64_t total = 0, failed = 0;
    int reported = 0;
    int max_report = 8;

    explicit Check(std::string n) : name(std::move(n)) {}

    template <typename... A>
    void fail(const char* fmt, A... args) {
        ++failed;
        if (reported < max_report) {
            std::printf("    MISMATCH [%s] ", name.c_str());
            std::printf(fmt, args...);
            std::printf("\n");
            ++reported;
        }
    }
    void expect(bool ok, const char* what) {
        ++total;
        if (!ok) fail("%s", what);
    }
    bool ok() const { return failed == 0; }
    void print() const {
        std::printf("  %-34s %s  (%llu checked, %llu mismatches)\n", name.c_str(), ok() ? "PASS" : "FAIL",
                    static_cast<unsigned long long>(total), static_cast<unsigned long long>(failed));
    }
};

}  // namespace mcc::test
