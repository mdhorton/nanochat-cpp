#include "nanochat/flags.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace nanochat {

Flags::Flags(int argc, char** argv, std::string description)
    : program_(argc > 0 ? argv[0] : "?")
    , description_(std::move(description)) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      help_ = true;
      continue;
    }
    if (!arg.starts_with("--"))
      throw std::invalid_argument("unexpected argument: " + arg);
    arg = arg.substr(2);
    if (auto eq = arg.find('='); eq != std::string::npos)
      values_[arg.substr(0, eq)] = arg.substr(eq + 1);
    else if (i + 1 < argc && !std::string_view(argv[i + 1]).starts_with("--"))
      values_[arg] = argv[++i];
    else
      values_[arg] = "true";
  }
}

const std::string* Flags::find(const std::string& name, const std::string& def, const std::string& help) {
  known_.insert(name);
  usage_ += "  --" + name + " (default: " + def + ")\n      " + help + "\n";
  auto it = values_.find(name);
  return it == values_.end() ? nullptr : &it->second;
}

std::string Flags::str(const std::string& name, const std::string& def, const std::string& help) {
  auto v = find(name, def, help);
  return v ? *v : def;
}

int64_t Flags::i64(const std::string& name, int64_t def, const std::string& help) {
  auto v = find(name, std::to_string(def), help);
  return v ? static_cast<int64_t>(std::stod(*v)) : def; // stod accepts 2e9
}

double Flags::f64(const std::string& name, double def, const std::string& help) {
  auto v = find(name, std::to_string(def), help);
  return v ? std::stod(*v) : def;
}

bool Flags::boolean(const std::string& name, bool def, const std::string& help) {
  auto v = find(name, def ? "true" : "false", help);
  if (!v)
    return def;
  if (*v == "true" || *v == "1")
    return true;
  if (*v == "false" || *v == "0")
    return false;
  throw std::invalid_argument("--" + name + " expects true/false, got " + *v);
}

void Flags::done() {
  if (help_) {
    std::cout << description_ << "\n\nusage: " << program_ << " [flags]\n" << usage_;
    std::exit(0);
  }
  for (const auto& [name, _] : values_)
    if (!known_.contains(name))
      throw std::invalid_argument("unknown flag --" + name + " (see --help)");
}

} // namespace nanochat
