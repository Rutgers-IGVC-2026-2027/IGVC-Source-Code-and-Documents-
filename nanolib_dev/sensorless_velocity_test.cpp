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

    // Standalone commissioning program; persistent setup stays in Plug & Drive.
    // C5-E FIR-v2039: 60E4/60E5 report per-feedback user-unit measurements.
    // Filename retained for compatibility; this test now requires Hall + encoder.
    auto read = [&](uint16_t index, uint8_t sub = 0) -> int64_t {
        const auto result = nanolib->readNumber(device, nlc::OdIndex(index, sub));
        if (result.hasError()) {
            std::ostringstream message;
            message << "Read 0x" << std::hex << index << ':' << unsigned(sub)
                    << ": " << result.getError();
            throw std::runtime_error(message.str());
        }
        return result.getResult();
    };
    auto signed32 = [&](uint16_t index, uint8_t sub = 0) -> int64_t {
        const auto value = static_cast<uint32_t>(read(index, sub));
        return value >= 0x80000000u ? int64_t(value) - 0x100000000LL : value;
    };
    auto write = [&](uint16_t index, int64_t value, unsigned bits) {
        const auto result = nanolib->writeNumber(device, value, nlc::OdIndex(index, 0), bits);
        if (result.hasError()) {
            std::ostringstream message;
            message << "Write 0x" << std::hex << index << ": " << result.getError();
            throw std::runtime_error(message.str());
        }
    };
    auto require = [](bool condition, const std::string& message) {
        if (!condition) throw std::runtime_error(message);
    };
    auto waitState = [&](uint16_t expected) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (keepRunning && std::chrono::steady_clock::now() < deadline) {
            const auto status = read(0x6041);
            require(!(status & 8), "Drive fault during state transition; no automatic reset.");
            if ((status & 0x006F) == expected) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        throw std::runtime_error("Enable transition interrupted or timed out.");
    };
    bool controlAttempted = false;
    int exitCode = 0;
    try {
        const auto firmware = nanolib->readString(device, nlc::OdIndex(0x100A, 0));
        require(!firmware.hasError(), "Cannot read firmware version.");
        std::cout << "Firmware: " << firmware.getResult() << '\n';
        const auto initialStatus = read(0x6041);
        require((initialStatus & 0x004F) == 0x0040,
                "Start with drive in Switch on disabled, without faults.");
        require(!(read(0x2300) & 1), "Stop the NanoJ program in Plug & Drive first.");
        require(read(0x3202) & 1, "Closed-loop configuration is not enabled.");
        require(read(0x3204, 0) == 3 && read(0x3204, 1) == 0x3380
                    && read(0x3204, 2) == 0x3390 && read(0x3204, 3) == 0x33A0,
                "Expected C5-E mapping: sensorless=1, Hall=2, incremental encoder=3.");
        int velocitySource = 0, commutationSource = 0;
        for (uint8_t sub = 1; sub <= 3; ++sub) {
            const auto mask = read(0x3203, sub);
            const auto increments = signed32(0x60E6, sub);
            const auto revolutions = read(0x60EB, sub);
            std::cout << "Feedback " << unsigned(sub) << " " << feedbackName(read(0x3204, sub))
                      << ": mask=" << mask << ", resolution=" << increments << '/' << revolutions << '\n';
            if (sub > 1) {
                require(increments != 0 && revolutions > 0,
                        "Hall/encoder resolution is missing; verify saved Plug & Drive configuration.");
                // Physical feedback priority is channel 2, then 3, then sensorless.
                if (!velocitySource && (mask & 2)) velocitySource = sub;
                if (!commutationSource && (mask & 4)) commutationSource = sub;
            }
        }
        require(velocitySource == 3,
                "Encoder must be the selected velocity source: enable encoder velocity and clear Hall velocity in Plug & Drive.");
        require(commutationSource == 2 || commutationSource == 3,
                "Select Hall or encoder commutation in Plug & Drive; sensorless-only commutation is not accepted.");
        std::cout << "Using encoder velocity; commutation source: "
                  << (commutationSource == 2 ? "Hall" : "encoder") << '\n';
        require(read(0x60A9) == 0x00B44700, "Set velocity user units to RPM in Plug & Drive.");
        for (const uint16_t index : {0x6096, 0x6097, 0x6091}) {
            const auto numerator = read(index, 1), denominator = read(index, 2);
            require(numerator > 0 && numerator == denominator,
                    "This direct-drive test requires 1:1 velocity/acceleration factors and gear ratio.");
        }
        for (uint8_t sub : {2, 3}) {
            require(read(0x60E8, sub) > 0 && read(0x60E8, sub) == read(0x60ED, sub),
                    "Hall and encoder feedback gear ratios must be 1:1 for this direct-drive test.");
        }
        const auto maxCurrent = read(0x2031), ratedCurrent = read(0x6075), peak = read(0x6073);
        std::cout << "Current limits: max=" << maxCurrent << " mA, rated=" << ratedCurrent
                  << " mA, peak=" << peak << " (0.1% rated)\n";
        require(maxCurrent > 0 && maxCurrent <= 3000 && ratedCurrent > 0 && ratedCurrent <= 3000
                    && peak > 0 && peak <= 1000,
                "Configure the agreed bench limits: <=3000 mA motor current, no peak allowance.");
        std::cout << "Max motor speed 0x6080=" << read(0x6080)
                  << " (controller units); profile limit=" << read(0x607F) << " rpm\n";
        const auto acceleration = read(0x6083), deceleration = read(0x6084);
        require(acceleration >= 25 && acceleration <= 100 && deceleration >= 25 && deceleration <= 100,
                "Set profile acceleration/deceleration to 25..100 rpm/s in Plug & Drive.");
        require(read(0x6086) == 0, "This test requires the linear motion profile (6086=0).");
        require(read(0x60C5) >= acceleration && read(0x60C6) >= deceleration,
                "Maximum acceleration/deceleration would restrict the configured ramps.");
        std::cout << "Hall/encoder velocity test. Enter positive RPM (1..350; start at 100): ";
        int rpm = 0;
        require(bool(std::cin >> rpm), "Invalid speed input.");
        require(rpm > 0 && rpm <= 350 && rpm <= read(0x607F), "Speed outside test/profile limits.");
        // Probe mandatory telemetry before any write.
        for (uint8_t sub : {2, 3}) {
            signed32(0x60E4, sub); signed32(0x60E5, sub);
        }
        signed32(0x606C);
        require(read(0x1001) == 0, "Clear the controller error before testing.");
        std::cout << "Runs for 10 seconds including ramp-up, then commands zero and disables.\n"
                  << "Uses saved encoder velocity and Hall/encoder commutation; no tuning writes.\n"
                  << "Use an unloaded secured motor, monitor the 24 V/3 A supply, and keep a hardware stop ready.\n"
                  << "Close Plug & Drive/other command sources. Type RUN to move: ";
        std::string confirmation;
        require(bool(std::cin >> confirmation) && confirmation == "RUN", "Motion cancelled.");
        require(keepRunning, "Interrupted before enable.");
        require((read(0x6041) & 0x004F) == 0x0040, "Drive state changed before enable.");
        controlAttempted = true; // Includes ambiguous/failed command writes in cleanup.
        write(0x60FF, 0, 32);
        write(0x6060, 3, 8); // Profile Velocity, NOT mode 2 / 6042.
        const auto modeDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (read(0x6061) != 3) {
            require(keepRunning && std::chrono::steady_clock::now() < modeDeadline, "Profile Velocity mode not acknowledged.");
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        require(keepRunning, "Interrupted before shutdown command.");
        write(0x6040, 0x0006, 16); waitState(0x0021);
        require(keepRunning, "Interrupted before switch on.");
        write(0x6040, 0x0007, 16); waitState(0x0023);
        require(keepRunning, "Interrupted before enable operation.");
        write(0x6040, 0x000F, 16); waitState(0x0027);
        require(keepRunning, "Interrupted before target velocity.");
        write(0x60FF, rpm, 32);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (keepRunning && std::chrono::steady_clock::now() < deadline) {
            const auto status = read(0x6041);
            require((status & 0x006F) == 0x0027, "Drive left Operation enabled or faulted.");
            require(read(0x1001) == 0, "Drive error register is nonzero.");
            require(read(0x6061) == 3, "Operation mode changed unexpectedly.");
            const auto velocity = signed32(0x60E5, 3), selectedVelocity = signed32(0x606C);
            require(velocity >= -50 && velocity <= 400 && selectedVelocity >= -50 && selectedVelocity <= 400,
                    "Unexpected direction or excessive reported velocity.");
            std::cout << "target=" << rpm << " demand=" << signed32(0x606B)
                      << " encoder velocity=" << velocity << " selected velocity=" << selectedVelocity
                      << " rpm; encoder position=" << signed32(0x60E4, 3)
                      << " user units; Hall velocity=" << signed32(0x60E5, 2)
                      << " rpm; Hall position=" << signed32(0x60E4, 2)
                      << " user units; status=0x" << std::hex << status << std::dec << '\n';
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    } catch (const std::exception& error) {
        std::cerr << "Test stopped: " << error.what() << '\n';
        exitCode = 1;
    }
    if (controlAttempted) {
        // Cleanup ignores keepRunning: Ctrl+C must still issue zero/disable.
        // No software cleanup can guarantee a stop after USB loss or process termination.
        bool stopped = false;
        try {
            write(0x60FF, 0, 32);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(16);
            int stationarySamples = 0;
            while (std::chrono::steady_clock::now() < deadline) {
                const auto velocity = signed32(0x606C);
                const auto encoderVelocity = signed32(0x60E5, 3);
                const auto demand = signed32(0x606B);
                if (velocity >= -5 && velocity <= 5 && encoderVelocity >= -5 && encoderVelocity <= 5 && demand == 0) ++stationarySamples;
                else stationarySamples = 0;
                if (stationarySamples >= 3) { stopped = true; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (!stopped) throw std::runtime_error("Zero-speed confirmation timed out.");
        } catch (const std::exception& error) {
            std::cerr << "Stop warning: " << error.what() << '\n';
            exitCode = 1;
        }
        // Attempt disable even if target-zero write or polling failed.
        try {
            write(0x6040, 0, 16);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while ((read(0x6041) & 0x004F) != 0x0040) {
                require(std::chrono::steady_clock::now() < deadline, "Disable not confirmed.");
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            std::cout << "Power stage disabled after stop attempt.\n";
        } catch (const std::exception& error) {
            std::cerr << "DISABLE FAILED: " << error.what() << ". Use the hardware stop.\n";
            exitCode = 1;
        }
    }
    const auto disconnected = nanolib->disconnectDevice(device);
    const auto removed = nanolib->removeDevice(device);
    const auto closed = nanolib->closeBusHardware(bus);
    if (disconnected.hasError() || removed.hasError() || closed.hasError()) {
        std::cerr << "Connection cleanup reported an error.\n";
        exitCode = 1;
    }
    return exitCode;
}
