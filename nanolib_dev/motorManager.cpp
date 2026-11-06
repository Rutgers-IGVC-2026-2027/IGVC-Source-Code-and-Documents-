#include "motorManager.hpp"
#include "accessor_factory.hpp"
#include "bus_hardware_options.hpp"
#include "nlc_callback.hpp"
#include <cstdint>

namespace {

class QuietScanCallback final : public nlc::NlcScanBusCallback {
public:
    nlc::ResultVoid callback(
        nlc::BusScanInfo,
        const std::vector<nlc::DeviceId>&,
        std::int32_t) override
    {
        return nlc::ResultVoid();
    }
};

} // namespace
//initiazes a nanolib accessor when motor manager is called for nanolib sdk fucntionality 
motorManager::motorManager()
    : nanolib_(getNanoLibAccessor())
{
}

motorConnectResult motorManager::connectMotors()
{
 
    if (nanolib_ == nullptr) {
        return {false, "Could not get the NanoLib accessor."};
    }

    if (motorBus_.has_value()) {
        return {false, "The manager already has an open motor bus."};
    }

    auto busResult = nanolib_->listAvailableBusHardware();
    if (busResult.hasError()) {
        return {false, "Bus discovery failed: " + busResult.getError()};
    }

    const std::string expectedBus =
        "BusHardwareId(USB Bus) [bus hardware: USB BUS, protocol: MODBUS VCP, "
        "hardware specifier: UsbRoot0, extra hardware specifier: UsbBusVcp]";

    const auto& buses = busResult.getResult();
    const nlc::BusHardwareId* selectedBus = nullptr;

    for (const auto& bus : buses) {
        if (bus.toString() == expectedBus) {
            if (selectedBus != nullptr) {
                return {false, "More than one bus matched the expected USB interface."};
            }
            selectedBus = &bus;
        }
    }

    if (selectedBus == nullptr) {
        return {false, "Expected Modbus VCP USB bus was not found."};
    }

    std::vector<nlc::DeviceHandle> registered;
    std::vector<nlc::DeviceHandle> connected;
    registered.reserve(2);
    connected.reserve(2);

    motorBus_.emplace(*selectedBus);
    nlc::BusHardwareOptions options;
    auto openResult = nanolib_->openBusHardwareWithProtocol(*motorBus_, options);

    if (openResult.hasError()) {
        motorBus_.reset();
        return {false, "USB bus open failed: " + openResult.getError()};
    }

    auto fail = [&](std::string message) -> motorConnectResult {
        for (const auto& handle : connected) {
            auto result = nanolib_->disconnectDevice(handle);
            if (result.hasError()) {
                message += " | Disconnect failed: " + result.getError();
            }
        }

        for (const auto& handle : registered) {
            auto result = nanolib_->removeDevice(handle);
            if (result.hasError()) {
                message += " | Remove device failed: " + result.getError();
            }
        }

        auto result = nanolib_->closeBusHardware(*motorBus_);
        if (result.hasError()) {
            message += " | Close bus failed: " + result.getError();
        } else {
            motorBus_.reset();
        }

        return {false, message};
    };

    QuietScanCallback callback;
    auto scanResult = nanolib_->scanDevices(*motorBus_, &callback);
    if (scanResult.hasError()) {
        return fail("Device scan failed: " + scanResult.getError());
    }

    const auto& scannedDevices = scanResult.getResult();
    const nlc::DeviceId* left = nullptr;
    const nlc::DeviceId* right = nullptr;

    for (const auto& device : scannedDevices) {
        const auto id = std::to_string(device.getDeviceId());

        if (id == leftDeviceId) {
            if (left != nullptr) {
                return fail("Duplicate left device ID: " + leftDeviceId);
            }
            left = &device;
        } else if (id == rightDeviceId) {
            if (right != nullptr) {
                return fail("Duplicate right device ID: " + rightDeviceId);
            }
            right = &device;
        }
    }

    if (left == nullptr) {
        return fail("Left motor was not found: " + leftDeviceId);
    }
    if (right == nullptr) {
        return fail("Right motor was not found: " + rightDeviceId);
    }

    auto connectOne = [&](const nlc::DeviceId& device,
                          const std::string& side) -> motorConnectResult {
        auto added = nanolib_->addDevice(device);
        if (added.hasError()) {
            return {false, side + " motor registration failed: " + added.getError()};
        }

        const auto handle = added.getResult();
        registered.push_back(handle);

        auto result = nanolib_->connectDevice(handle);
        if (result.hasError()) {
            return {false, side + " motor connection failed: " + result.getError()};
        }

        connected.push_back(handle);
        return {true, {}};
    };

    auto leftResult = connectOne(*left, "Left");
    if (!leftResult.ok) {
        return fail(leftResult.error);
    }

    auto rightResult = connectOne(*right, "Right");
    if (!rightResult.ok) {
        return fail(rightResult.error);
    }

    devices_.swap(connected);
    return {true, {}};}

    nlc::DeviceHandle motorManager:: handleFor(motorSide side) const
{
    if (devices_.size() != 2) {
        throw std::logic_error("Connect both motors before accessing feedback.");
    }

    return side == motorSide::Left ? devices_[0] : devices_[1];
}

std::int64_t motorManager::readNumber(motorSide side, nlc::OdIndex address)
{
    const auto result = nanolib_->readNumber(handleFor(side), address);

    if (result.hasError()) {
        throw std::runtime_error("Object dictionary read failed: " +
                                 result.getError());
    }

    return result.getResult();
}

void motorManager::writeNumber(motorSide side, nlc::OdIndex address,
                               std::int64_t value, unsigned int bits)
{
    const auto result =
        nanolib_->writeNumber(handleFor(side), value, address, bits);

    if (result.hasError()) {
        throw std::runtime_error("Object dictionary write failed: " +
                                 result.getError());
    }
}

