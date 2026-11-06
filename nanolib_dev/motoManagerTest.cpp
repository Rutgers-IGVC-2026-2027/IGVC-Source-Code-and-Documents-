#include "motorManager.hpp"
using namespace std;


int main(){
motorManager manager;

auto result=manager.connectMotors();

cout << "Connected: " << std::boolalpha << result.ok
          << " | Error: " << result.error << endl;
    return 0;
}
