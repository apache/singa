#include "singa/utils/timer.h"
#include <iostream>
#include <unistd.h>

int main(){
    singa::Timer t;
    double t1 = t.elapsed();
    std::cout << "t1 = " << t1 << std::endl;
    sleep(1);
    double t2 = t.elapsed();
    std::cout << "t2 = " << t2 << std::endl;
    std::cout << "delta = " << t2 - t1 << " ms" << std::endl;
    
    return 0;

}
