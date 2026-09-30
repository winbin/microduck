//! Which servo family the robot is actually fitted with.
//!
//! One enum in front of two backends. The choice is a property of the *hardware*, so it comes
//! from `robotd.toml` and is made once, at `open_bus`, rather than being discovered — nothing
//! on the wire can tell the two protocols apart, because a bus wired for one and configured
//! for the other simply times out.
//!
//! It is a hand-written enum rather than a `Box<dyn RobotIo>`, and that is not a style
//! preference. The startup path calls `check_registers`, `missing_servos`,
//! `adopt_replacement`, `present_positions` and `interpolate_to`, and none of those are on
//! [`RobotIo`] — they are the bus's own business, deliberately not part of the seam the
//! control loop is written against. `Safety` is generic over the backend too, so a trait
//! object would not reach it either. Both backends keep their inherent methods and this
//! forwards to them, one line each, which is what makes the compiler the thing that keeps
//! the two in step.

use std::time::Duration;

use crate::bus::DynamixelIo;
use crate::bus_feetech::FeetechIo;
use crate::io::{ImuStale, JointTargets, Result, RobotIo, Sensors, SlowSensors};
use crate::model::NUM_JOINTS;

/// The robot's servos, whichever family is fitted.
///
/// Both variants are boxed, which is not symmetry for its own sake: a backend carries a
/// controller handle and per-tick cached state, so they are a few hundred bytes each and
/// leaving them inline would size every `AnyBus` as the larger of the two — for a value there
/// is exactly one of in the process. The indirection costs one load against bus transactions
/// that are already hundreds of microseconds of serial I/O.
pub enum AnyBus {
    /// Dynamixel XL330 on Protocol 2.0 — the family the policies were trained on.
    Dynamixel(Box<DynamixelIo>),
    /// Feetech SCS/STS, the HD-1910 — same robot, different wire.
    Feetech(Box<FeetechIo>),
}

impl AnyBus {
    /// Which family this is, for a log line or a health report. Named rather than derived so
    /// a reader of the journal can tell what a board is running without knowing the enum.
    pub fn protocol(&self) -> &'static str {
        match self {
            Self::Dynamixel(_) => "dynamixel2",
            Self::Feetech(_) => "feetech-sts",
        }
    }

    /// Verify — and where safe, correct — the registers the control loop depends on.
    pub fn check_registers(&mut self) -> Result<usize> {
        match self {
            Self::Dynamixel(io) => io.check_registers(),
            Self::Feetech(io) => io.check_registers(),
        }
    }

    /// The expected servo IDs that do not answer a ping.
    pub fn missing_servos(&mut self) -> Result<Vec<u8>> {
        match self {
            Self::Dynamixel(io) => io.missing_servos(),
            Self::Feetech(io) => io.missing_servos(),
        }
    }

    /// Flash a factory-fresh servo as the joint that is missing.
    pub fn adopt_replacement(&mut self, id: u8) -> Result<bool> {
        match self {
            Self::Dynamixel(io) => io.adopt_replacement(id),
            Self::Feetech(io) => io.adopt_replacement(id),
        }
    }

    /// Present positions only — the lighter startup read.
    pub fn present_positions(&mut self) -> Result<[f64; NUM_JOINTS]> {
        match self {
            Self::Dynamixel(io) => io.present_positions(),
            Self::Feetech(io) => io.present_positions(),
        }
    }

    /// Torque on or off, every joint.
    pub fn set_torque(&mut self, on: bool) -> Result<()> {
        match self {
            // Type-qualified: both backends carry this method on the trait *and* inherently,
            // and the point of this one is the inherent pair.
            Self::Dynamixel(io) => DynamixelIo::set_torque(io, on),
            Self::Feetech(io) => FeetechIo::set_torque(io, on),
        }
    }

    /// Ramp every joint to `target` over `duration`. Only ever called by an explicit `init`.
    pub fn interpolate_to(
        &mut self,
        target: &[f64; NUM_JOINTS],
        duration: Duration,
        step: Duration,
    ) -> Result<()> {
        match self {
            Self::Dynamixel(io) => io.interpolate_to(target, duration, step),
            Self::Feetech(io) => io.interpolate_to(target, duration, step),
        }
    }
}

/// The seam itself. Every method here is one the control loop uses, and every one of them is
/// also an inherent method above where the two backends disagree about a signature.
impl RobotIo for AnyBus {
    fn read(&mut self) -> Result<Sensors> {
        match self {
            Self::Dynamixel(io) => io.read(),
            Self::Feetech(io) => io.read(),
        }
    }

    fn write(&mut self, targets: &JointTargets) -> Result<()> {
        match self {
            Self::Dynamixel(io) => io.write(targets),
            Self::Feetech(io) => io.write(targets),
        }
    }

    fn set_gain(&mut self, kp: u16) -> Result<()> {
        match self {
            Self::Dynamixel(io) => io.set_gain(kp),
            Self::Feetech(io) => io.set_gain(kp),
        }
    }

    fn set_torque(&mut self, on: bool) -> Result<()> {
        // The inherent method, so the two impls cannot accidentally recurse into this one.
        AnyBus::set_torque(self, on)
    }

    fn reboot(&mut self, id: u8) -> Result<()> {
        match self {
            Self::Dynamixel(io) => io.reboot(id),
            Self::Feetech(io) => io.reboot(id),
        }
    }

    fn slow_sensors(&mut self) -> Result<SlowSensors> {
        match self {
            Self::Dynamixel(io) => io.slow_sensors(),
            Self::Feetech(io) => io.slow_sensors(),
        }
    }

    fn imu_stale(&self) -> ImuStale {
        match self {
            Self::Dynamixel(io) => io.imu_stale(),
            Self::Feetech(io) => io.imu_stale(),
        }
    }

    fn imu_ready(&self) -> bool {
        match self {
            Self::Dynamixel(io) => io.imu_ready(),
            Self::Feetech(io) => io.imu_ready(),
        }
    }
}
