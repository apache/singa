%module driver
%include "std_vector.i"
%include "std_string.i"
%include "argcargv.i"
%apply (int ARGC, char **ARGV) { (int argc, char **argv)  }
%{
#include "../include/singa/driver.h"
%}

namespace singa{
using std::vector;
class Driver{
public:
void Train(bool resume, const std::string job_conf);
void Init(int argc, char **argv);
};
}

