#ifndef ELECTRICAL_LOAD_HPP
#define ELECTRICAL_LOAD_HPP

// Electrical power readings for one component.
class eLoad {
public:
    eLoad(double voltage = 0.0, double current = 0.0);

    // Accept volts and amps from manual input or a future hardware interface.
    // Throws std::invalid_argument for negative/non-finite readings or power overflow.
    void setReadings(double voltage, double current);

    double getVoltage() const;
    double getCurrent() const;
    double getPowerDraw() const; // Watts: voltage * current.

private:
    double voltage = 0.0;
    double current = 0.0;
};

#endif
