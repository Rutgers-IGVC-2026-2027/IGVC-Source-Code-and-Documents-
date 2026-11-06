#pragma once 

#include <limits>
#include <stdexcept>
#include <sstream>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <optional>
#include "accessor_factory.hpp"
#include "bus_hardware_options.hpp"
#include "nano_lib_accessor.hpp"
#include "nano_lib_hw_strings.hpp"
#include "nlc_callback.hpp"
#include "nlc_log_level.hpp"
#include "od_index.hpp"

struct motorConnectResult{
    bool ok;
    std:: string error;
};
class motorManager {

public:

enum class motorSide{ Left, Right};

motorManager();

[[nodiscard]] motorConnectResult connectMotors();


 

 //object motor help write

    private:
   
    nlc::NanoLibAccessor* nanolib_;

    std::optional<nlc::BusHardwareId> motorBus_;

    
    std::vector<nlc::DeviceHandle> devices_;
    
    std::string leftDeviceId = "525289";
    
    std::string rightDeviceId = "459753";

    nlc::DeviceHandle handleFor(motorSide side) const;

    std::int64_t readNumber(motorSide side, nlc::OdIndex address);

    void writeNumber(motorSide side, nlc::OdIndex address,
                        std::int64_t value, unsigned int bits);

};

// automatically configure 2 motors over usb 

//helper write fucntion

// helper read fucntion 
