//! The robot, as data.
//!
//! One variant — **alpha** — because that is the only robot that exists. Every shipped
//! policy is `alpha_*`; v1/v1.5/v1.6 are history. A second revision becomes a second set
//! of tables, which is honest until there is a second robot to generalise from.
//!
//! The numeric values here are lifted from `microduck_runtime`'s `motor.rs`, where they
//! were measured against hardware rather than derived. Re-deriving them from a datasheet
//! is exactly the kind of change that looks right and walks wrong.

/// Left leg (5) · neck/head/mouth (5) · right leg (5).
pub const NUM_JOINTS: usize = 15;

/// Dynamixel IDs, indexed as [`JOINT_NAMES`].
pub const JOINT_IDS: [u8; NUM_JOINTS] = [
    20, 21, 22, 23, 24, // left leg
    30, 31, 32, 33, 34, // neck, head, mouth
    10, 11, 12, 13, 14, // right leg
];

/// Joint names, from the protocol crate — the wire indexes `joints` and `targets`
/// positionally, so that order and this one cannot be allowed to drift apart. The
/// assertion below is what makes "cannot" true.
pub use duck_ipc_proto::JOINT_NAMES;

const _: () = assert!(JOINT_NAMES.len() == NUM_JOINTS);

/// The mouth is absent from every alpha policy — they are all 61-D observation, 14-action,
/// and the action vector skips this index. Named so that omission is deliberate rather
/// than an off-by-one someone has to rediscover.
pub const MOUTH_INDEX: usize = 9;

/// Home pose. The trunk sits ~5 mm further forward than the v1.5 pose so the CoM is over
/// the ankle axis; the old pose biased the robot backwards.
///
/// Must match `HOME_FRAME` in the training env — a policy is trained against these angles
/// and observes joint positions *relative* to them, so a discrepancy here is a constant
/// offset on 14 observation slots.
pub const DEFAULT_POSITION: [f64; NUM_JOINTS] = [
    0.0,     // left_hip_yaw
    -0.0873, // left_hip_roll
    -0.4579, // left_hip_pitch
    -0.0049, // left_knee
    0.4530,  // left_ankle
    0.3491,  // neck_pitch
    0.3491,  // head_pitch
    0.0,     // head_yaw
    0.0,     // head_roll
    0.0,     // mouth
    0.0,     // right_hip_yaw
    0.0873,  // right_hip_roll
    0.4579,  // right_hip_pitch
    0.0049,  // right_knee
    -0.4530, // right_ankle
];

/// Mouth travel, radians: closed and fully open. The alpha reuses the v1.6 range,
/// −5°..+30°, from `microduck_runtime`'s `variant.rs`.
///
/// The mouth is not part of any policy — every alpha network is 14 actions with this joint
/// skipped — so these two numbers and [`mouth_target`] are the whole of mouth control.
pub const MOUTH_CLOSED: f64 = -5.0 * std::f64::consts::PI / 180.0;
pub const MOUTH_OPEN: f64 = 30.0 * std::f64::consts::PI / 180.0;

/// Joint angle for a mouth opening fraction. 0 is closed, 1 is fully open; anything outside
/// is clamped rather than fed to a servo as an out-of-travel target.
pub fn mouth_target(open: f64) -> f64 {
    let open = if open.is_finite() {
        open.clamp(0.0, 1.0)
    } else {
        0.0
    };
    MOUTH_CLOSED + open * (MOUTH_OPEN - MOUTH_CLOSED)
}

/// The `imu_to_dxl` v2 board's Dynamixel ID. It rides the motor bus and is read in the
/// same transaction as the servos ([`crate::bus`]).
pub const IMU_DXL_ID: u8 = 200;

pub const BAUD_RATE: u32 = 1_000_000;

/// What a servo answers as out of the box: ID 1 at 57 600 baud. Both are deliberately unused
/// on this bus — no joint is ID 1 and nothing runs at that speed — which is what lets a
/// replacement be told apart from every servo already fitted ([`crate::bus`]).
pub const FACTORY_ID: u8 = 1;
pub const FACTORY_BAUD_RATE: u32 = 57_600;

/// EEPROM registers asserted (and corrected) at startup.
///
/// `return_delay_time` is the load-bearing one: the XL330 ships at 250, which is 500 µs of
/// turnaround *per device*. Across 16 devices that is 8 ms per tick — 40% of a 20 ms budget
/// — spent waiting for servos to get around to answering. The rest are here because the
/// runtime found them worth pinning.
///
/// `shutdown = 52` is `0b110100` — overload, electrical shock, overheating — with the
/// input-voltage bit **clear**, where the factory's 53 sets it. That bit is what clears torque
/// once the supply passes the servo's `Max Voltage Limit`, which nothing here writes and which
/// therefore stays at its default 7.0 V. A charged 2S pack sits above that, so the clear bit is
/// the reason fifteen servos do not latch themselves off a fully charged battery. Read as
/// "latches on input-voltage faults" it says the opposite of what it does.
pub const EXPECTED_REGISTERS: &[(&str, u8)] = &[
    ("return_delay_time", 0),
    ("baud_rate", 3), // 3 = 1 Mbps, must agree with BAUD_RATE
    ("pwm_slope", 255),
    ("shutdown", 52),
];

// ── Feetech SCS/STS — the HD-1910 ────────────────────────────────────────────
//
// The same fifteen joints, driven by a different servo family on the same wire. Nothing
// above this line changes: `JOINT_IDS`, `DEFAULT_POSITION` and every policy are statements
// about the *robot*, not about who makes its motors. What changes is the register map, the
// scaling, and the fact that the IMU board answers at a different address in a longer block.
//
// The numbers below come from Feetech's SCS/STS control table for the magnetic-encoder
// series, and match `rustypot`'s own `sts3215` definition address for address — that
// agreement is what makes it safe to read a raw block by address here instead of by name.

/// `id` — writable without unlocking on a factory servo, and where the replacement path
/// writes the missing joint's number.
pub const FT_ID_ADDR: u8 = 5;

/// `baudrate`. Feetech counts *down* from 1 Mbps: 0 is 1 Mbps, 1 is 500 k, 2 is 250 k, and
/// so on to 7 (19 200). The factory value is already 0, so unlike the XL330 there is no
/// speed to change before a fresh servo can be adopted.
pub const FT_BAUD_ADDR: u8 = 6;

/// `response_status_level`. At 1 every instruction is acknowledged; at 0 only PING and READ
/// are, so every write this code issues would wait out the full timeout and fail. Asserted
/// rather than written — see [`crate::bus_feetech::FeetechIo::check_registers`].
pub const FT_RESPONSE_STATUS_ADDR: u8 = 8;

/// `phase`. Read once at startup: BIT2 chooses the present-speed unit (see
/// [`FT_PHASE_SPEED_UNIT_BIT`]), and BIT0/BIT7 the direction this bus counts in.
pub const FT_PHASE_ADDR: u8 = 18;

/// `p_coefficient`. A `u8`, so 0–254, and in the EEPROM region (5–39) — which is why
/// [`crate::bus_feetech::FeetechIo::set_gain`] writes it without unlocking.
pub const FT_P_GAIN_ADDR: u8 = 21;
pub const FT_D_GAIN_ADDR: u8 = 22;
pub const FT_I_GAIN_ADDR: u8 = 23;

/// `mode`. HD-1910 leaves the factory in 4 — Feetech's own name for it is "pure position
/// PD", the mode sim2real is trained against — and [`FT_MODE_POSITION`] is what this robot
/// runs. 0 also works; anything else is looked at by a person rather than overwritten.
pub const FT_MODE_ADDR: u8 = 33;

/// `torque_enable`: 0 off, 1 on, 2 damping, 128 mid-position calibration.
pub const FT_TORQUE_ENABLE_ADDR: u8 = 40;

/// `goal_position`, `i16`, 4096 counts per revolution.
pub const FT_GOAL_POSITION_ADDR: u8 = 42;

/// `lock`. Write 1 to lock, 0 to unlock. A locked write to an EEPROM address is *accepted
/// and not persisted* — that is the documented meaning, and it is what makes a locked gain
/// write behave like the XL330's RAM register. The factory value is 1 (locked).
pub const FT_LOCK_ADDR: u8 = 55;

/// Start of the block read every tick, and its length.
///
/// Fifteen bytes covers, per servo: `present_position` (56), `present_speed` (58),
/// `present_load` (60), `present_voltage` (62), `present_temperature` (63), torque feedback
/// (64), `status` (65), `moving` (66), two reserved bytes, and `present_current` (69).
///
/// The IMU board answers at the *same* address with its own fifteen bytes, which is what
/// lets one `sync_read` fetch all sixteen devices — the same shape the Dynamixel bus used,
/// at a different address and one byte longer.
pub const FT_READ_ADDR: u8 = 56;
pub const FT_READ_LEN: u8 = 15;

/// Offsets inside that block. Named rather than inlined because a wrong one hands a joint
/// its neighbour's value, which reads as a wiring fault rather than a bug.
pub const FT_OFF_POSITION: usize = 0;
pub const FT_OFF_SPEED: usize = 2;
pub const FT_OFF_LOAD: usize = 4;
pub const FT_OFF_VOLTAGE: usize = 6;
pub const FT_OFF_TEMPERATURE: usize = 7;
pub const FT_OFF_STATUS: usize = 9;
pub const FT_OFF_CURRENT: usize = 13;

/// `status` at 65 — the hardware-error bits, the counterpart of the XL330's
/// `hardware_error_status` and masked by `unloading_condition` at 19.
pub const FT_STATUS_ADDR: u8 = 65;

/// `present_current` — reported for the state stream, which wants load, not sign.
pub const FT_PRESENT_CURRENT_ADDR: u8 = 69;

/// `END` at address 2: 0 is little-endian, which is what every multi-byte field above is
/// read as. A one-byte read asserted at startup rather than assumed: the whole encoding
/// series is documented little-endian, and a device that disagreed would produce plausible
/// joint angles from swapped bytes.
pub const FT_ENDIAN_ADDR: u8 = 2;
pub const FT_ENDIAN_LITTLE: u8 = 0;

/// The value [`FT_BAUD_ADDR`] must hold for [`BAUD_RATE`] to be the speed this bus talks at.
pub const FT_BAUD_RATE_CODE: u8 = 0;

/// The value [`FT_RESPONSE_STATUS_ADDR`] must hold.
pub const FT_RESPONSE_STATUS_LEVEL: u8 = 1;

/// The value [`FT_MODE_ADDR`] should hold — reported, never written.
pub const FT_MODE_POSITION: u8 = 4;

/// Bit of [`FT_PHASE_ADDR`] that selects the present-speed unit.
///
/// **This is the one number on this bus that can silently ruin a gait.** The speed register
/// is the same either way; the bit decides whether one count means 1 step/s or 50 steps/s,
/// so reading it with the wrong scale multiplies every joint velocity in the observation
/// vector by fifty. A policy tolerates that just well enough to walk badly, which is the
/// failure this robot is least able to see. `bus.speed_unit` in `robotd.toml` overrides the
/// bit for a bench that measures otherwise.
pub const FT_PHASE_SPEED_UNIT_BIT: u8 = 2;

/// Counts per revolution. The same 4096 as the XL330, and the same centre at 2048 — which
/// is why `DEFAULT_POSITION` and every trained policy carry over unchanged.
pub const FT_COUNTS_PER_REV: f64 = 4096.0;

/// Radians per count, and — the same number — rad/s per speed count when the phase bit
/// selects the 1 step/s unit. One step is one count, so the two cannot differ.
pub const FT_RAD_PER_COUNT: f64 = 2.0 * std::f64::consts::PI / FT_COUNTS_PER_REV;

/// Rad/s per speed count when the phase bit selects the 50 steps/s unit.
pub const FT_RAD_PER_COUNT_X50: f64 = 50.0 * FT_RAD_PER_COUNT;

/// Milliamps per `present_current` count.
pub const FT_MA_PER_CURRENT_UNIT: f64 = 6.5;

/// Volts per `present_voltage` count.
pub const FT_VOLTS_PER_COUNT: f64 = 0.1;

/// What a Feetech servo answers as out of the box: ID 1, already at 1 Mbps.
///
/// The ID is deliberately unused on this bus and is the same number the XL330 ships with,
/// which is what lets the replacement path be shared between the two backends. The speed
/// being the same is the difference that matters: adopting a fresh servo needs no port
/// reopen here, so [`crate::bus_feetech`] has no factory baud rate at all.
pub const FT_FACTORY_ID: u8 = 1;

/// Registers [`crate::bus_feetech::FeetechIo::check_registers`] verifies, as
/// `(name, address, required)`.
///
/// Both are load-bearing rather than cosmetic. A servo not at 1 Mbps cannot be talking to
/// us at all, so a wrong value here means the bus is in a state this code has no business
/// commanding; and at response level 0 the servo answers reads and pings and nothing else,
/// so every write — torque, gain, goal position — times out. The XL330's `shutdown` mask
/// and `pwm_slope` have no counterpart: overload protection on these servos lives in
/// `unloading_condition` (19) and friends, which the vendor tool sets once and this code
/// only reports.
pub const FT_EXPECTED_REGISTERS: &[(&str, u8, u8)] = &[
    ("baud_rate", FT_BAUD_ADDR, FT_BAUD_RATE_CODE),
    (
        "response_status_level",
        FT_RESPONSE_STATUS_ADDR,
        FT_RESPONSE_STATUS_LEVEL,
    ),
];

/// Feetech position register → radians, in the servo's own sense.
///
/// The same map as the XL330: 4096 counts per revolution, 2048 at the centre. Masked to
/// fifteen bits rather than read as a two's-complement `i16` because the STS encoding makes
/// bit 15 a *sign* — `sign_magnitude(15)` in rustypot's own definition — and a servo in a
/// position mode never sets it: travel is 0..4095. Reading it as `i16` turns a count the
/// servo cannot send into a large negative angle.
pub fn feetech_position_rad(raw: u16) -> f64 {
    let counts = (raw & 0x7FFF) as f64;
    2.0 * std::f64::consts::PI * counts / FT_COUNTS_PER_REV - std::f64::consts::PI
}

/// The inverse, clamped to the servo's travel so a target outside it becomes an end stop
/// rather than a wrapped one on the far side of the joint.
pub fn feetech_position_counts(rad: f64) -> u16 {
    let counts =
        (FT_COUNTS_PER_REV * (std::f64::consts::PI + rad) / (2.0 * std::f64::consts::PI)).round();
    if !counts.is_finite() {
        return 0;
    }
    counts.clamp(0.0, FT_COUNTS_PER_REV - 1.0) as u16
}

/// A sign-magnitude `u16` as its signed value: bit 15 is the sign, bits 0–14 the magnitude.
///
/// Not two's complement. Feetech uses this encoding for speed, load and current, and
/// reading one as `i16` makes every negative value a large positive one — a joint reported
/// as spinning flat out rather than reversing.
pub fn sign_magnitude(raw: u16) -> i32 {
    let magnitude = (raw & 0x7FFF) as i32;
    if raw & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Index of a joint by name. Linear scan over 15 entries, used at startup and in tests.
pub fn joint_index(name: &str) -> Option<usize> {
    JOINT_NAMES.iter().position(|n| *n == name)
}

// ── battery ──────────────────────────────────────────────────────────────────
//
// There is no fuel gauge and no ADC. The only measurement available is what the servos
// report as their own supply (`crate::bus::DynamixelIo::bus_voltage`), which is the pack
// seen through the bus — so it sags under load and recovers when the robot stands still.
// That is why the span below is *usable-under-load*, not the cell chemistry's range.
//
// The span is a statement about the *rail*, and it holds because that rail is the pack: the
// servos are fed the 2S battery, not a regulated 5 V. A servo on a bench supply reads its own
// supply voltage like any other, so 5 V lands under `BATTERY_EMPTY_V` and maps to 0% — the
// mapping working as defined rather than a fault to chase. Anything that regulates the servo
// rail has to move these two constants with it.

/// Off a full charge, under load. NP-F550, 2S Li-ion.
pub const BATTERY_FULL_V: f64 = 8.2;

/// The sag floor: below this the robot starts struggling, well before the pack's own
/// protection trips. Empty for our purposes, not empty for the cells'.
pub const BATTERY_EMPTY_V: f64 = 6.6;

/// Fraction of a pack, 0–100, for a bus voltage.
///
/// Linear, and the numbers come from `microduck_runtime`'s `check_battery`, where they were
/// arrived at by running a duck flat. It lives here rather than in `robotd` so there is one
/// mapping: the prototype had it in a CLI *and* re-derived in the app, which is how two
/// screens end up disagreeing about the same pack.
///
/// A non-finite or non-positive reading is 0 — those mean "no answer from the bus", and the
/// caller is expected to report that as unknown rather than to display this number.
pub fn battery_percent(volts: f64) -> f64 {
    if !volts.is_finite() || volts <= 0.0 {
        return 0.0;
    }
    ((volts - BATTERY_EMPTY_V) / (BATTERY_FULL_V - BATTERY_EMPTY_V)).clamp(0.0, 1.0) * 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three tables are indexed by the same integer everywhere in the crate. If they
    /// ever diverge in length, every lookup silently reads the wrong joint.
    #[test]
    fn tables_agree_on_length() {
        assert_eq!(JOINT_IDS.len(), NUM_JOINTS);
        assert_eq!(JOINT_NAMES.len(), NUM_JOINTS);
        assert_eq!(DEFAULT_POSITION.len(), NUM_JOINTS);
    }

    /// A duplicated Dynamixel ID makes a `sync_read` return blocks that cannot be matched
    /// back to joints, and a `sync_write` command two joints at once. Both fail in ways
    /// that look like a wiring fault.
    #[test]
    fn ids_are_unique() {
        let mut seen = JOINT_IDS;
        seen.sort_unstable();
        seen.windows(2)
            .for_each(|w| assert_ne!(w[0], w[1], "duplicate Dynamixel ID {}", w[0]));
    }

    /// The IMU board shares the bus with the servos, so its ID must not collide with one.
    #[test]
    fn imu_id_does_not_collide_with_a_joint() {
        assert!(!JOINT_IDS.contains(&IMU_DXL_ID));
    }

    /// The replacement path finds a new servo by the ID it ships with. If a joint ever took
    /// ID 1, a fresh servo would be indistinguishable from it — and flashing "the missing
    /// joint" onto ID 1 would re-address a servo that was never missing.
    #[test]
    fn factory_defaults_are_unused_on_the_bus() {
        assert!(!JOINT_IDS.contains(&FACTORY_ID));
        assert_ne!(IMU_DXL_ID, FACTORY_ID);
        assert_ne!(FACTORY_BAUD_RATE, BAUD_RATE);
    }

    /// `shutdown` is the one register whose *bits* are the decision rather than the number:
    /// the input-voltage bit is cleared on purpose, and that is what lets the pack's range run
    /// across a servo rated to 6.0 V. The factory default is 53 — one bit away — so a later
    /// edit that "restores the default" should meet a test rather than a comment.
    #[test]
    fn the_shutdown_mask_clears_the_input_voltage_bit() {
        let want = EXPECTED_REGISTERS
            .iter()
            .find(|(n, _)| *n == "shutdown")
            .map(|&(_, v)| v)
            .expect("shutdown is an expected register");
        const OVERLOAD: u8 = 1 << 5;
        const ELECTRICAL_SHOCK: u8 = 1 << 4;
        const OVERHEATING: u8 = 1 << 2;
        const INPUT_VOLTAGE: u8 = 1 << 0;
        assert_eq!(
            want,
            OVERLOAD | ELECTRICAL_SHOCK | OVERHEATING,
            "the three faults worth latching on, with input-voltage clear"
        );
        assert_eq!(want & INPUT_VOLTAGE, 0);
    }

    /// `MOUTH_INDEX` is used to skip a slot when mapping 14 policy actions onto 15 joints.
    /// Pointing it at the wrong joint would shift every action after it by one.
    #[test]
    fn mouth_index_names_the_mouth() {
        assert_eq!(JOINT_NAMES[MOUTH_INDEX], "mouth");
        assert_eq!(joint_index("mouth"), Some(MOUTH_INDEX));
    }

    /// The ends of the span and the middle of it. Getting the direction wrong here would
    /// report a full pack as flat, which is the kind of thing nobody double-checks.
    #[test]
    fn battery_percent_spans_the_usable_range() {
        assert_eq!(battery_percent(BATTERY_FULL_V), 100.0);
        assert_eq!(battery_percent(BATTERY_EMPTY_V), 0.0);
        assert!((battery_percent(7.4) - 50.0).abs() < 0.001);
    }

    /// Voltages outside the span are ordinary — a fresh pack reads over 8.2 V off the
    /// charger, and a robot being run into the ground reads under 6.6 V. Neither may
    /// produce a percentage outside 0–100 for a caller to display.
    #[test]
    fn battery_percent_clamps_rather_than_extrapolating() {
        assert_eq!(battery_percent(9.5), 100.0);
        assert_eq!(battery_percent(5.0), 0.0);
    }

    /// A bus that did not answer arrives here as 0.0, and NaN is what a mean over an empty
    /// set produces. Both must be 0 rather than a wild number a caller might print.
    #[test]
    fn battery_percent_treats_no_reading_as_zero() {
        assert_eq!(battery_percent(0.0), 0.0);
        assert_eq!(battery_percent(f64::NAN), 0.0);
        assert_eq!(battery_percent(-1.0), 0.0);
    }

    /// The mouth range is the prototype's: −5° closed, +30° open. A fraction outside 0..1
    /// (or a NaN from a broken client) must clamp rather than command a servo past travel.
    #[test]
    fn mouth_target_spans_the_prototype_range() {
        assert!((mouth_target(0.0) - (-5.0f64.to_radians())).abs() < 1e-12);
        assert!((mouth_target(1.0) - 30.0f64.to_radians()).abs() < 1e-12);
        assert_eq!(mouth_target(-3.0), mouth_target(0.0));
        assert_eq!(mouth_target(7.0), mouth_target(1.0));
        assert_eq!(mouth_target(f64::NAN), mouth_target(0.0));
    }

    /// The legs are mirrored: the roll/pitch/ankle pairs are equal and opposite. A sign
    /// typo in the home pose is invisible by inspection and makes the robot stand crooked.
    #[test]
    fn home_pose_legs_are_mirrored() {
        for (left, right) in [
            ("left_hip_roll", "right_hip_roll"),
            ("left_hip_pitch", "right_hip_pitch"),
            ("left_knee", "right_knee"),
            ("left_ankle", "right_ankle"),
        ] {
            let l = DEFAULT_POSITION[joint_index(left).unwrap()];
            let r = DEFAULT_POSITION[joint_index(right).unwrap()];
            assert!(
                (l + r).abs() < 1e-9,
                "{left} ({l}) and {right} ({r}) should be equal and opposite"
            );
        }
    }

    /// The whole reason fifteen trained policies carry over to a different servo family:
    /// 4096 counts, 2048 at the centre, the same map the XL330 uses. If this ever drifts,
    /// every joint is offset by a constant and the robot stands wrong by inspection only.
    #[test]
    fn feetech_position_matches_the_xl330_map() {
        assert!((feetech_position_rad(2048) - 0.0).abs() < 1e-12);
        assert!((feetech_position_rad(1024) + std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((feetech_position_rad(3072) - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        for raw in [0u16, 1, 1024, 2048, 3072, 4095] {
            assert_eq!(feetech_position_counts(feetech_position_rad(raw)), raw);
        }
    }

    /// Bit 15 is a sign in the STS encoding, and a servo in a position mode never sets it —
    /// travel is 0..4095. Read as a two's-complement `i16` instead, `0x8000 | 2048` becomes
    /// −30720 counts: a joint on the far side of its travel rather than at its centre.
    #[test]
    fn feetech_position_is_masked_to_fifteen_bits() {
        assert_eq!(
            feetech_position_rad(0x8000 | 2048),
            feetech_position_rad(2048)
        );
        assert_eq!(
            feetech_position_rad(0x8000 | 4095),
            feetech_position_rad(4095)
        );
        assert_eq!(feetech_position_rad(0x8000), feetech_position_rad(0));
        assert_eq!(
            (0x8000u16 | 2048) as i16,
            -30720,
            "what reading it unsigned would give"
        );
    }

    /// A target past the end of travel has to become an end stop. Wrapping instead would
    /// send a joint to the far side of its range, which on a leg is a fall.
    #[test]
    fn feetech_position_counts_clamp_rather_than_wrap() {
        assert_eq!(feetech_position_counts(std::f64::consts::PI), 4095);
        assert_eq!(feetech_position_counts(10.0), 4095);
        assert_eq!(feetech_position_counts(-10.0), 0);
        assert_eq!(feetech_position_counts(f64::NAN), 0);
    }

    /// Sign-magnitude, not two's complement: the failure mode of getting it wrong is a
    /// reversing joint reported as one spinning flat out, which the policy reads as a
    /// robot in trouble rather than a servo going the other way.
    #[test]
    fn sign_magnitude_decodes_both_halves() {
        assert_eq!(sign_magnitude(0), 0);
        assert_eq!(sign_magnitude(1), 1);
        assert_eq!(sign_magnitude(0x7FFF), 32767);
        assert_eq!(sign_magnitude(0x8001), -1);
        assert_eq!(sign_magnitude(0xFFFF), -32767);
        assert_eq!(sign_magnitude(0x8000), 0, "negative zero");
    }

    /// The two candidate speed units differ by exactly the factor the phase bit selects.
    /// Picking the wrong one scales every observed joint velocity by fifty — the mistake
    /// `bus.speed_unit` exists to make fixable without a code change.
    #[test]
    fn the_two_speed_units_differ_by_fifty() {
        assert!((FT_RAD_PER_COUNT_X50 / FT_RAD_PER_COUNT - 50.0).abs() < 1e-12);
        // 0.0146 RPM and 0.732 RPM are the datasheet's own statement of the same two units.
        let rpm = FT_RAD_PER_COUNT * 60.0 / (2.0 * std::f64::consts::PI);
        assert!(
            (rpm - 0.0146).abs() < 5e-5,
            "1 step/s should read 0.0146 RPM, got {rpm}"
        );
    }

    /// The check table and the constants have to say the same thing. `bus.port`-style drift
    /// between a table and the numbers it encodes is how a register ends up asserted against
    /// the wrong address, and this test is cheaper than the bench session that finds it.
    #[test]
    fn feetech_expected_registers_agree_with_the_constants() {
        for (name, addr, want) in FT_EXPECTED_REGISTERS {
            match *name {
                "baud_rate" => {
                    assert_eq!(*addr, FT_BAUD_ADDR);
                    assert_eq!(*want, FT_BAUD_RATE_CODE);
                }
                "response_status_level" => {
                    assert_eq!(*addr, FT_RESPONSE_STATUS_ADDR);
                    assert_eq!(*want, FT_RESPONSE_STATUS_LEVEL);
                }
                other => panic!("unhandled register {other}"),
            }
        }
    }

    /// The block the tick reads has to cover the last field it parses. If the offsets and
    /// the length ever disagree, every joint silently gets its neighbour's current.
    #[test]
    fn the_feetech_block_covers_every_field_it_parses() {
        assert_eq!(FT_READ_ADDR + FT_READ_LEN, FT_STATUS_ADDR + 6);
        const { assert!(FT_OFF_CURRENT + 2 == FT_READ_LEN as usize) };
        const { assert!(FT_OFF_TEMPERATURE < FT_READ_LEN as usize) };
        const { assert!(FT_OFF_VOLTAGE < FT_READ_LEN as usize) };
    }

    /// The IMU board answers at the same address with a fifteen-byte block of its own, so
    /// the block has to be long enough for the twelve bytes the decoder consumes.
    #[test]
    fn the_feetech_block_carries_the_imu_block() {
        assert!(FT_READ_LEN as usize >= crate::imu::IMU_BLOCK_LEN);
    }
}
