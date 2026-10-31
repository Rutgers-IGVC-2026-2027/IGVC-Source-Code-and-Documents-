#include "electricalLoad.hpp"

#include <cmath>
#include <stdexcept>

eLoad::eLoad(double voltage, double current) {
    setReadings(voltage, current);
}

void eLoad::setReadings(double voltage, double current) {
    if (!std::isfinite(voltage) || voltage < 0.0 ||
        !std::isfinite(current) || current < 0.0 ||
        !std::isfinite(voltage * current)) {
        throw std::invalid_argument(
            "load voltage and current must be finite, nonnegative, and produce finite power");
    }

    this->voltage = voltage;
    this->current = current;
}

double eLoad::getVoltage() const {
    return voltage;
}

double eLoad::getCurrent() const {
    return current;
}

double eLoad::getPowerDraw() const {
    return voltage * current;
}
