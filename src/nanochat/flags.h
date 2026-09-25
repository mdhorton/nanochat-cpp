// Minimal command line flags: --name=value or --name value.
// Declare each flag with a getter, then call done() to handle --help and reject unknown flags.
#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>

namespace nanochat {

    class Flags {
    public:
        Flags(int argc, char **argv, std::string description);

        std::string str(const std::string &name, const std::string &def, const std::string &help);
        int64_t i64(const std::string &name, int64_t def, const std::string &help);
        double f64(const std::string &name, double def, const std::string &help);
        bool boolean(const std::string &name, bool def, const std::string &help);

        void done();

    private:
        const std::string *find(const std::string &name, const std::string &def, const std::string &help);

        std::string program_, description_, usage_;
        std::map<std::string, std::string> values_;
        std::set<std::string> known_;
        bool help_ = false;
    };

} // namespace nanochat
