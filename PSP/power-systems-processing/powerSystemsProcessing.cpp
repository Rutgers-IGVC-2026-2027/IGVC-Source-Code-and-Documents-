#include "powerSystemsProcessing.hpp"

#include <cmath>
#include <ostream>
#include <stdexcept>

namespace {
bool validCharge(double value) {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

bool finiteNonnegative(double value) {
    return std::isfinite(value) && value >= 0.0;
}

bool finitePositive(double value) {
    return std::isfinite(value) && value > 0.0;
}

template <typename Signal>
void displaySignal(const char* name, const Signal& signal, std::ostream& output) {
    output << name << ": " << signal.getVoltage() << " V, "
           << signal.getCurrent() << " A, " << signal.getPowerDraw() << " W\n";
}
}

powerSystemsProcessing::powerSystemsProcessing(const Limits& limits)
    : batteryActions(limits.lowBatteryStatus, limits.criticalBatteryStatus),
      safetyActions(limits) {}

double powerSystemsProcessing::totalPowerDraw(const MeasuredValues& values) {
    double total = values.lidar.getPowerDraw() + values.camera.getPowerDraw()
                 + values.computer.getPowerDraw();
    for (const auto& motor : values.motors) {
        total += motor.getPowerDraw();
    }
    // Battery input power is not added: that would count the same loads twice.
    return total;
}

void powerSystemsProcessing::MeasuredValuesDisplay::display(
    const MeasuredValues& values, std::ostream& output) const {
    if (finiteNonnegative(values.batteryVoltage) &&
        finiteNonnegative(values.batteryCurrent) &&
        std::isfinite(values.batteryVoltage * values.batteryCurrent)) {
        output << "Battery: " << values.batteryVoltage << " V, "
               << values.batteryCurrent << " A, "
               << values.batteryVoltage * values.batteryCurrent << " W\n";
    } else {
        output << "Battery: missing or invalid electrical readings\n";
    }
    if (validCharge(values.batteryStatus)) {
        output << "Battery charge: " << values.batteryStatus * 100.0 << "%\n";
    } else {
        output << "Battery charge: missing or invalid\n";
    }
    displaySignal("LiDAR", values.lidar, output);
    displaySignal("Camera", values.camera, output);
    displaySignal("Computer", values.computer, output);
    for (std::size_t index = 0; index < values.motors.size(); ++index) {
        output << "Motor " << index + 1 << " ";
        displaySignal("readings", values.motors[index], output);
    }
    output << "Total component power draw: " << totalPowerDraw(values) << " W\n";
}

powerSystemsProcessing::BatteryActions::BatteryActions(
    double lowBatteryStatus, double criticalBatteryStatus)
    : lowBatteryStatus(lowBatteryStatus), criticalBatteryStatus(criticalBatteryStatus) {
    if (!validCharge(lowBatteryStatus) || !validCharge(criticalBatteryStatus) ||
        criticalBatteryStatus >= lowBatteryStatus) {
        throw std::invalid_argument("Battery thresholds must satisfy 0 <= critical < low <= 1");
    }
}

powerSystemsProcessing::Decision powerSystemsProcessing::BatteryActions::evaluate(
    double batteryStatus) const {
    if (!validCharge(batteryStatus)) {
        return {Action::RequestStop, "Battery charge is missing or invalid"};
    }
    if (batteryStatus <= criticalBatteryStatus) {
        return {Action::RequestStop, "Battery charge is critical"};
    }
    if (batteryStatus <= lowBatteryStatus) {
        return {Action::ReducePower, "Battery charge is low"};
    }
    return {Action::ContinueOperation, "Battery charge is sufficient"};
}

powerSystemsProcessing::SafetyActions::SafetyActions(const Limits& limits)
    : limits(limits) {
    if (!finitePositive(limits.minimumBatteryVoltage) ||
        !finitePositive(limits.maximumBatteryVoltage) ||
        limits.minimumBatteryVoltage >= limits.maximumBatteryVoltage ||
        !finitePositive(limits.maximumBatteryCurrent) ||
        !finitePositive(limits.maximumMotorCurrent) ||
        !finitePositive(limits.maximumTotalPower)) {
        throw std::invalid_argument("Safety limits must be finite and positive, with minimum voltage < maximum voltage");
    }
}

powerSystemsProcessing::Decision powerSystemsProcessing::SafetyActions::evaluate(
    const MeasuredValues& values) const {
    if (!finiteNonnegative(values.batteryVoltage) ||
        !finiteNonnegative(values.batteryCurrent) ||
        !std::isfinite(values.batteryVoltage * values.batteryCurrent) ||
        !validCharge(values.batteryStatus) || !std::isfinite(totalPowerDraw(values))) {
        return {Action::RequestStop, "Measurements are missing or invalid"};
    }
    // Voltage endpoints are allowed; exceeding a current/power maximum trips.
    if (values.batteryVoltage < limits.minimumBatteryVoltage ||
        values.batteryVoltage > limits.maximumBatteryVoltage) {
        return {Action::RequestStop, "Battery voltage is outside its limits"};
    }
    if (values.batteryCurrent > limits.maximumBatteryCurrent) {
        return {Action::RequestStop, "Battery current exceeds its limit"};
    }
    for (const auto& motor : values.motors) {
        if (motor.getCurrent() > limits.maximumMotorCurrent) {
            return {Action::RequestStop, "A motor current exceeds its limit"};
        }
    }
    if (totalPowerDraw(values) > limits.maximumTotalPower) {
        return {Action::RequestStop, "Total component power exceeds its limit"};
    }
    return {Action::ContinueOperation, "Measurements are within safety limits"};
}

void powerSystemsProcessing::displayMeasurements(
    const MeasuredValues& values, std::ostream& output) const {
    measuredValuesDisplay.display(values, output);
}

powerSystemsProcessing::Decision powerSystemsProcessing::determineAction(
    const MeasuredValues& values) const {
    const Decision safetyDecision = safetyActions.evaluate(values);
    if (safetyDecision.action != Action::ContinueOperation) {
        return safetyDecision;
    }
    return batteryActions.evaluate(values.batteryStatus);
}
