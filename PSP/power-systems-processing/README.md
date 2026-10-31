# Power Systems Processing

IGVC power systems processing project in C++ for evaluating robot power measurements and returning
decisions to continue operation, reduce power, or request a stop.

## Redundant class cleanup

The redundant `cameraSignal`, `computerSignal`, `lidarSignal`, and `motorSignal`
classes have been replaced with one shared `eLoad` class. Each component still
has its own readings, while validation and power calculations now live in
`electricalLoad.cpp`. The obsolete component files and unused `manualSignal`
helper have been removed, and the VS Code build now compiles the shared class.
The existing processing tests pass with this replacement.

## Project files

- `powerSystemsProcessing.hpp` and `.cpp`: measurement handling and decision logic.
- `electricalLoad.hpp` and `.cpp`: the shared `eLoad` class for component measurements.
- `tests/powerSystemsProcessingTest.cpp`: assertion-based tests.
- `../.vscode/`: shared build and debug configurations for macOS.

## The `powerSystemsProcessing` class

`powerSystemsProcessing` brings the robot's electrical measurements together and
evaluates what action the rest of the system should take. Its constructor accepts
a `Limits` object containing the low and critical battery-charge thresholds,
allowed battery-voltage range, and maximum battery current, individual motor
current, and total component power.

Each call to `determineAction()` evaluates a `MeasuredValues` snapshot. This
structure contains battery voltage, current, and charge; the LiDAR, camera, and
computer objects; and a vector of motor objects. Charge
is represented as a fraction, so `0.8` means 80%.

The class organizes its work into three nested helper classes:

- `SafetyActions` checks measurement validity, voltage
  range, current limits, and total power limit. A failed check requests a stop.
- `BatteryActions` checks battery charge after the safety checks pass. Charge at
  or below the critical threshold requests a stop; charge above critical but at
  or below the low threshold requests reduced power. Higher charge allows normal
  operation.
- `MeasuredValuesDisplay` formats the readings and total power for an output
  stream, such as the terminal or a string used in a test.

`determineAction()` returns a `Decision` containing an `action` and a readable
`reason`. The possible actions are `ContinueOperation`, `ReducePower`, and
`RequestStop`. Safety checks take priority over the battery-charge recommendation.
The caller is responsible for acting on the decision. The processing class
evaluates each supplied snapshot independently and does not remember a previous
stop request or directly operate hardware.

`totalPowerDraw()` adds the power drawn by the LiDAR, camera, computer, and all
motors. Battery input power is displayed separately and is excluded from that
sum to avoid counting the same loads twice.

## Electrical load class

`eLoad`, declared in `electricalLoad.hpp`, represents the electrical load of a
LiDAR, camera, computer, or motor. Each object stores
voltage in volts and current in amps, and calculates power in watts using
`power = voltage * current`.

Use `setReadings(voltage, current)` to update a component, then `getVoltage()`,
`getCurrent()`, and `getPowerDraw()` to retrieve its measurements. Construction and
updates reject negative or non-finite readings, as well as values whose product
is not finite, by throwing `std::invalid_argument`. Default component readings
are zero volts and zero amps.

This class gives `MeasuredValues` a consistent way to represent the
robot's electrical loads. Each motor has its own object in the `motors` vector,
allowing the processor to check individual motor currents and sum their power.
Sensor data collection and device control belong in separate interfaces.

## Current tests and the planned dynamic test

`tests/powerSystemsProcessingTest.cpp` currently checks specific measurement
snapshots using `assert`. It sets up normal readings, changes selected values,
and checks the returned decisions. The cases cover battery-charge thresholds,
voltage boundaries, current and power limits, safety-check priority, invalid
measurements, invalid limit configurations, and formatted measurement output.
These tests run without a simulation clock or ROS 2 messages.

The planned **dynamic test** will evaluate a sequence of measurements as simulated
time advances. At each step, a simulation will update battery charge and component
loads, assemble a new `MeasuredValues` snapshot, and pass it to the processor.
The test will check both the decision at each step and when the action changes.

For example, using the current test's illustrative charge thresholds and keeping
all other measurements within their limits:

| Point in a discharge scenario | Battery charge | Expected action |
| --- | --- | --- |
| Initial state | 80% | `ContinueOperation` |
| Charge reaches the low threshold | 20% | `ReducePower` |
| Charge reaches the critical threshold | 10% | `RequestStop` |

Additional scenarios will vary motor loads and introduce voltage drops. A simulated
controller will apply the returned decisions to the next simulation step, making it possible to test how reduced loads affect later
measurements. Recording simulated time, readings, actions, and reasons will make
each transition reviewable. This dynamic test is planned; the existing test file
currently provides the fixed measurement cases described above.

## Build and run

Use a C++17 compiler. From the repository root on macOS:

```sh
cd power-systems-processing
clang++ -std=c++17 -g -I. electricalLoad.cpp powerSystemsProcessing.cpp tests/powerSystemsProcessingTest.cpp -o tests/powerSystemsProcessingTest
./tests/powerSystemsProcessingTest
```

Successful tests currently finish without printing anything. A failed assertion
reports the failure and stops the program. Keep assertions enabled when running
tests; do not compile them with `NDEBUG` defined.

In VS Code, open the repository root folder and use the Microsoft C/C++ extension:

- **Cmd+Shift+B** builds the test.
- **F5**, with **Debug power systems test** selected, builds and debugs it.

The VS Code task lists the current source files explicitly. Add new `.cpp` files
to `../.vscode/tasks.json` when expanding the project. Header files are included by
the source files and do not need separate compiler entries.

The limits used in the tests are illustrative values, not robot hardware limits.

## Planned ROS 2 integration and dynamic simulation

Future development will extend the current fixed-value tests into a ROS 2
simulation where `MeasuredValues` changes over time. These features are planned
and are not implemented yet.

1. **Connect the processing code to ROS 2.** Create a package and a processing
   node that receives simulated measurements, updates `MeasuredValues`, calls
   `determineAction()`, and publishes the resulting action and reason. Keep the
   existing processing class usable independently for unit tests.
2. **Generate changing measurements.** Add a simulation node that publishes
   battery voltage, battery current, battery charge, and voltage/current readings
   for the LiDAR, camera, computer, and each motor at a configurable interval.
   Add functions to update battery charge based on current draw and elapsed time,
   vary motor loads, and change component activity. Keep `batteryStatus` expressed
   as a fraction from 0.0 to 1.0.
3. **Add repeatable scenarios.** Simulate normal operation, gradual battery
   discharge, sudden motor load increases, and voltage drops. Provide controls to
   start, pause, and reset a scenario, with a fixed random seed when adding measurement noise so failures can be reproduced.
4. **Simulate responses to decisions.** Add a simulated controller that reduces
   component loads after `ReducePower` and stops simulated motor activity after
   `RequestStop`. Feed the resulting measurements back into the processing node
   to observe how decisions affect subsequent readings.
5. **Make the simulation configurable and observable.** Expose initial readings,
   battery capacity, update interval, and processing limits as ROS 2 parameters.
   Log or plot battery charge, component power, total power draw, and action
   changes over time.
6. **Test behavior over time.** Add integration tests that check transitions from
   `ContinueOperation` to `ReducePower` and `RequestStop`. Extend measurement
   handling with timestamps and tests for missing, delayed, and invalid readings.

The first milestone is a simulated battery that gradually discharges while the
processing node reports the expected action changes at the configured low and
critical charge thresholds.

## Repository

Clone the [Rutgers IGVC Power System Processing repository](https://github.com/Rutgers-IGVC-2026-2027/Power-System-Processing-)
and use the build instructions above:

```sh
git clone https://github.com/Rutgers-IGVC-2026-2027/Power-System-Processing-.git
cd Power-System-Processing-
```

Review `git status` before committing. The `.gitignore` excludes compiled output,
macOS metadata, and local environment files while keeping source files and the
shared VS Code configuration.
