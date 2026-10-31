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

#include "accessor_factory.hpp"
#include "bus_hardware_options.hpp"
#include "nano_lib_accessor.hpp"
#include "nano_lib_hw_strings.hpp"
#include "nlc_callback.hpp"
#include "nlc_log_level.hpp"
#include "od_index.hpp"

namespace {

volatile std::sig_atomic_t keepRunning = 1;

class ScanCallback final : public nlc::NlcScanBusCallback {
public:
    nlc::ResultVoid callback(nlc::BusScanInfo info,
                             const std::vector<nlc::DeviceId>&,
                             int32_t) override {
        if (info == nlc::BusScanInfo::Start) {
            std::cout << "Scanning for controllers...\n";
        } else if (info == nlc::BusScanInfo::Finished) {
            std::cout << "Scan finished.\n";
        }
        return nlc::ResultVoid();
    }
};

const char* feedbackName(int64_t mapping) {
    switch (mapping) {
    case 0x3380:
        return "sensorless";
    case 0x3390:
        return "Hall";
    case 0x33A0:
        return "incremental A/B/I 1";
    case 0x33A1:
        return "incremental A/B/I 2";
    case 0x33A2:
        return "incremental A/B/I 3";
    default:
        return "other/unknown";
    }
}

template <typename Item>
std::size_t chooseItem(const std::vector<Item>& items, const std::string& prompt) {
    if (items.size() == 1) {
        return 0;
    }

    std::cout << prompt;
    std::size_t choice = 0;
    while (!(std::cin >> choice) || choice < 1 || choice > items.size()) {
        if (std::cin.eof()) { throw std::runtime_error("Input closed"); }
        std::cin.clear();
        std::cin.ignore(10000, '\n');
        std::cout << "Enter a number from 1 to " << items.size() << ": ";
    }
    return choice - 1;
}

} // namespace

int main() {
    std::signal(SIGINT, [](int) { keepRunning = 0; });
    std::signal(SIGTERM, [](int) { keepRunning = 0; });
    nlc::NanoLibAccessor* nanolib = getNanoLibAccessor();
    nanolib->setLoggingLevel(nlc::LogLevel::Off);

    auto busResult = nanolib->listAvailableBusHardware();
    if (busResult.hasError()) {
        std::cerr << "USB bus scan failed: " << busResult.getError() << '\n';
        return 1;
    }

    const auto& discoveredBuses = busResult.getResult();
    if (discoveredBuses.empty()) {
        std::cerr << "NanoLib found no bus hardware at all.\n"
                  << "Make sure the original example is closed and all NanoLib .so files are installed.\n";
        return 1;
    }

    std::cout << "NanoLib discovered these bus interfaces:\n";
    for (const auto& bus : discoveredBuses) {
        std::cout << "  " << bus.toString() << '\n';
    }

    std::vector<nlc::BusHardwareId> usbBuses;
    for (const auto& bus : discoveredBuses) {
        if (bus.getProtocol() == nlc::BUS_HARDWARE_ID_PROTOCOL_MODBUS_VCP ||
            bus.getProtocol() == nlc::BUS_HARDWARE_ID_PROTOCOL_MSC) {
            usbBuses.push_back(bus);
        }
    }

    if (usbBuses.empty()) {
        std::cerr << "No MODBUS VCP or MSC USB interface was present in the list above.\n"
                  << "Check controller power, USB cable, Linux permissions, and close the original example.\n";
        return 1;
    }

    std::cout << "USB interfaces:\n";
    for (std::size_t index = 0; index < usbBuses.size(); ++index) {
        std::cout << "  " << index + 1 << ") " << usbBuses[index].toString() << '\n';
    }
    const auto& bus = usbBuses[chooseItem(usbBuses, "Select USB interface: ")];

    nlc::BusHardwareOptions options;
    auto openResult = nanolib->openBusHardwareWithProtocol(bus, options);
    if (openResult.hasError()) {
        std::cerr << "Could not open USB interface: " << openResult.getError() << '\n';
        return 1;
    }

    ScanCallback scanCallback;
    std::vector<nlc::DeviceId> devices;
    std::string lastScanError;
    for (int attempt = 1; attempt <= 3 && devices.empty(); ++attempt) {
        std::cout << "Waiting for the USB interface (attempt " << attempt << "/3)...\n";
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto deviceResult = nanolib->scanDevices(bus, &scanCallback);
        if (deviceResult.hasError()) {
            lastScanError = deviceResult.getError();
        } else {
            for (const auto& foundDevice : deviceResult.getResult()) {
                devices.emplace_back(foundDevice);
            }
        }
    }

    if (devices.empty()) {
        std::cerr << "No controller found";
        if (!lastScanError.empty()) {
            std::cerr << ": " << lastScanError;
        }
        std::cerr << "\nVerify that you selected the working MSC (USB BUS) entry.\n";
        nanolib->closeBusHardware(bus);
        return 1;
    }

    std::cout << "Controllers:\n";
    for (std::size_t index = 0; index < devices.size(); ++index) {
        std::cout << "  " << index + 1 << ") " << devices[index].toString() << '\n';
    }
    const auto& deviceId = devices[chooseItem(devices, "Select controller: ")];

    auto addResult = nanolib->addDevice(deviceId);
    if (addResult.hasError()) {
        std::cerr << "Could not add controller: " << addResult.getError() << '\n';
        nanolib->closeBusHardware(bus);
        return 1;
    }
    const nlc::DeviceHandle device = addResult.getResult();

    auto connectResult = nanolib->connectDevice(device);
    if (connectResult.hasError()) {
        std::cerr << "Could not connect to controller: " << connectResult.getError() << '\n';
        nanolib->removeDevice(device);
        nanolib->closeBusHardware(bus);
        return 1;
    }


    // C5-E FIR-v2039: 60E4/60E5 are per-feedback USER units, not raw pulses.
    auto printValue = [&](uint16_t index, uint8_t sub, bool signed32 = false) {
        const auto result = nanolib->readNumber(device, nlc::OdIndex(index, sub));
        if (result.hasError()) {
            std::cout << "ERROR(" << result.getError() << ')';
        } else {
            const auto raw = static_cast<uint32_t>(result.getResult());
            const int64_t value = signed32 && raw >= 0x80000000u
                ? static_cast<int64_t>(raw) - 0x100000000LL : result.getResult();
            std::cout << value;
        }
    };
    std::vector<uint8_t> feedbacks;
    const auto count = nanolib->readNumber(device, nlc::OdIndex(0x3204, 0));
    if (!count.hasError() && count.getResult() >= 1 && count.getResult() <= 3) {
        for (uint8_t sub = 1; sub <= count.getResult(); ++sub) {
            feedbacks.push_back(sub);
            const auto mapping = nanolib->readNumber(device, nlc::OdIndex(0x3204, sub));
            std::cout << "Feedback " << unsigned(sub) << ": ";
            if (mapping.hasError()) std::cout << "mapping ERROR: " << mapping.getError();
            else std::cout << feedbackName(mapping.getResult());
            std::cout << ", selection mask="; printValue(0x3203, sub);
            std::cout << ", increments="; printValue(0x60E6, sub, true);
            std::cout << ", motor revolutions="; printValue(0x60EB, sub);
            std::cout << '\n';
        }
    } else {
        std::cerr << "Cannot determine C5-E feedback count; no channel assumptions made.\n";
    }
    std::cout << "Selection bits: 1=position, 2=velocity, 4=commutation.\n"
              << "Zero encoder increments means feedback is unused/disconnected.\n"
              << "Encoder electrical configuration 0x2059="; printValue(0x2059, 0);
    std::cout << "\nPosition unit 0x60A8 (decimal encoding)="; printValue(0x60A8, 0);
    std::cout << ", velocity unit 0x60A9="; printValue(0x60A9, 0);
    std::cout << "\nREAD-ONLY: no configuration, enable, or motion writes.\n"
              << "With the power stage disabled, rotate the shaft by hand when safe.\n"
              << "Per-feedback position/velocity are configured user units, NOT raw counts.\n"
              << "Sampling sequentially about five times/second; Ctrl+C stops.\n";
    while (keepRunning) {
        for (const auto sub : feedbacks) {
            std::cout << "feedback " << unsigned(sub) << " position=";
            printValue(0x60E4, sub, true);
            std::cout << " velocity="; printValue(0x60E5, sub, true);
            std::cout << " | ";
        }
        std::cout << "selected position="; printValue(0x6064, 0, true);
        std::cout << " velocity="; printValue(0x606C, 0, true);
        std::cout << '\n';
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << "\nStopping and disconnecting...\n";
    nanolib->disconnectDevice(device);
    nanolib->removeDevice(device);
    nanolib->closeBusHardware(bus);
    return 0;
}
