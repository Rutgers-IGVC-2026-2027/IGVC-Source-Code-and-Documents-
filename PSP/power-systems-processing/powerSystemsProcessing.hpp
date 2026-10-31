#ifndef POWER_SYSTEMS_PROCESSING_HPP
#define POWER_SYSTEMS_PROCESSING_HPP

#include "electricalLoad.hpp"

#include <iosfwd>
#include <vector>

class powerSystemsProcessing {
public:
    enum class Action { ContinueOperation, ReducePower, RequestStop };

    struct Decision {
        Action action;
        const char* reason;
    };

    struct MeasuredValues {
        // Negative defaults mean no battery measurement has been supplied.
        double batteryVoltage = -1.0; // Volts.
        double batteryCurrent = -1.0; // Amps drawn from the battery.
        double batteryStatus = -1.0;  // Charge fraction: 0.0 to 1.0.
        eLoad lidar;
        eLoad camera;
        eLoad computer;
        std::vector<eLoad> motors; // One entry per motor.
    };

    // Supply limits appropriate to your robot; there are no hardware defaults.
    struct Limits {
        double lowBatteryStatus;
        double criticalBatteryStatus;
        double minimumBatteryVoltage;
        double maximumBatteryVoltage;
        double maximumBatteryCurrent;
        double maximumMotorCurrent; // Applies to each motor separately.
        double maximumTotalPower;   // Sum of component power draws, in watts.
    };

    class MeasuredValuesDisplay {
    public:
        void display(const MeasuredValues& values, std::ostream& output) const;
    };

    class BatteryActions {
    public:
        BatteryActions(double lowBatteryStatus, double criticalBatteryStatus);
        Decision evaluate(double batteryStatus) const;

    private:
        double lowBatteryStatus;
        double criticalBatteryStatus;
    };

    class SafetyActions {
    public:
        explicit SafetyActions(const Limits& limits);
        Decision evaluate(const MeasuredValues& values) const;

    private:
        Limits limits;
    };

    explicit powerSystemsProcessing(const Limits& limits);

    void displayMeasurements(const MeasuredValues& values, std::ostream& output) const;
    // Safety checks take priority over battery charge decisions.
    // Returns a request for the caller to handle; does not control hardware.
    // Evaluate again whenever measurements change.
    Decision determineAction(const MeasuredValues& values) const;
    static double totalPowerDraw(const MeasuredValues& values);

private:
    MeasuredValuesDisplay measuredValuesDisplay;
    BatteryActions batteryActions;
    SafetyActions safetyActions;
};

#endif
