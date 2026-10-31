#include  "powerSystemsProcessing.hpp"

#include <cassert>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

int main() {
    using Processing = powerSystemsProcessing;
    using Action = Processing::Action;
    // Illustrative test values only, not recommended robot hardware limits.
    const Processing::Limits limits{0.2, 0.1, 10.0, 15.0, 20.0, 5.0, 100.0};
    Processing processing(limits);
    Processing::MeasuredValues readings;
    assert(processing.determineAction(readings).action == Action::RequestStop);

    readings.batteryVoltage = 12.0;
    readings.batteryCurrent = 4.0;
    readings.batteryStatus = 0.8;
    readings.lidar.setReadings(12.0, 0.5);
    readings.camera.setReadings(5.0, 1.0);
    readings.computer.setReadings(12.0, 1.0);
    readings.motors.emplace_back(12.0, 1.0);
    readings.motors.emplace_back(12.0, 1.0);
    assert(Processing::totalPowerDraw(readings) == 47.0);
    assert(processing.determineAction(readings).action == Action::ContinueOperation);
    const auto normal = readings; // make measurements default to this value every test

    readings.batteryStatus = 0.2;
    assert(processing.determineAction(readings).action == Action::ReducePower);
    readings.batteryStatus = 0.1;
    assert(processing.determineAction(readings).action == Action::RequestStop);
    readings.batteryStatus = 0.0;
    assert(processing.determineAction(readings).action == Action::RequestStop);
    readings.batteryStatus = 1.0;
    assert(processing.determineAction(readings).action == Action::ContinueOperation);

    // A voltage limit violation must override the low-battery recommendation.
    readings.batteryStatus = 0.2;
    readings.batteryVoltage = 9.9;
    auto decision = processing.determineAction(readings);
    assert(decision.action == Action::RequestStop);
    assert(std::string(decision.reason) == "Battery voltage is outside its limits");

    readings = normal;
    for (double voltage : {10.0, 15.0}) {
        readings.batteryVoltage = voltage;
        assert(processing.determineAction(readings).action == Action::ContinueOperation);
    }
    for (double voltage : {9.9, 15.1}) {
        readings.batteryVoltage = voltage;
        assert(processing.determineAction(readings).action == Action::RequestStop);
    }
    readings = normal;
    readings.batteryCurrent = 20.0;
    assert(processing.determineAction(readings).action == Action::ContinueOperation);
    readings.batteryCurrent = 20.1;
    assert(processing.determineAction(readings).action == Action::RequestStop);

    readings = normal;
    readings.motors[1].setReadings(1.0, 5.0);
    assert(processing.determineAction(readings).action == Action::ContinueOperation);
    readings.motors[1].setReadings(1.0, 5.1);
    assert(processing.determineAction(readings).action == Action::RequestStop);

    readings = normal;
    readings.computer.setReadings(65.0, 1.0);
    assert(Processing::totalPowerDraw(readings) == 100.0);
    assert(processing.determineAction(readings).action == Action::ContinueOperation);
    readings.computer.setReadings(66.0, 1.0);
    assert(processing.determineAction(readings).action == Action::RequestStop);

    readings = normal;
    for (double invalid : {-1.0, 1.1, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {

        readings.batteryStatus = invalid;
        assert(processing.determineAction(readings).action == Action::RequestStop);
    }
    readings = normal;
    readings.batteryCurrent = std::numeric_limits<double>::quiet_NaN();
    assert(processing.determineAction(readings).action == Action::RequestStop);
    readings = normal;
    readings.camera.setReadings(std::numeric_limits<double>::max(), 1.0);
    readings.computer.setReadings(std::numeric_limits<double>::max(), 1.0);
    assert(processing.determineAction(readings).action == Action::RequestStop);

    for (int index = 0; index < 4; ++index) {
        auto invalidLimits = limits;
        if (index == 0) invalidLimits.criticalBatteryStatus = limits.lowBatteryStatus;
        if (index == 1) invalidLimits.maximumBatteryVoltage = limits.minimumBatteryVoltage;
        if (index == 2) invalidLimits.maximumMotorCurrent = -1.0;
        if (index == 3) invalidLimits.maximumTotalPower = std::numeric_limits<double>::infinity();
        bool rejected = false;
        try { Processing invalid(invalidLimits); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }

    std::ostringstream output;
    processing.displayMeasurements(normal, output);
    const auto text = output.str();
    assert(text.find("Battery charge: 80%") != std::string::npos);
    assert(text.find("LiDAR: 12 V, 0.5 A, 6 W") != std::string::npos);
    assert(text.find("Motor 2 readings: 12 V, 1 A, 12 W") != std::string::npos);
    assert(text.find("Total component power draw: 47 W") != std::string::npos);
}
