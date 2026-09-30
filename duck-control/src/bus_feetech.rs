//! The Feetech SCS/STS bus — the HD-1910 servos.
//!
//! The same shape as [`crate::bus`]: one combined `sync_read` per tick covering the IMU board
//! and all fifteen servos, one `sync_write` of goal positions, and a startup pass that
//! verifies what the loop depends on. What differs is everything below the shape.
//!
//! **The wire.** `FF FF ID LEN INSTR … CHK`, one's-complement checksum, 1 Mbps, half duplex.
//! Protocol 1 framing in `rustypot`'s terms, and *only* that framing: the fast sync read the
//! Dynamixel bus uses (0x8A) is a Protocol 2 instruction this family does not implement, so
//! `sync_read` (0x82) is the only one there is and every device answers for itself.
//!
//! **The block.** Address 56, fifteen bytes, versus 124 and twelve. Fifteen rather than
//! twelve because position, speed, load, voltage, temperature, torque feedback, status,
//! `moving` and current are contiguous here — which means voltage and thermals arrive *in*
//! the tick's transaction instead of costing a second one every second, and the IMU board's
//! twelve bytes ride along in the first fifteen-byte answer with three bytes of its own to
//! spare.
//!
//! **The sense.** An HD-1910 counts up where an XL330 counts down, joint for joint. That is
//! a property of the hardware, not of the protocol, and it is why [`FeetechIo::open`] takes
//! a direction per joint: applied on the way in *and* on the way out, so `DEFAULT_POSITION`,
//! every policy and every observation keep the sense they were trained in.
//!
//! **The gains.** Feetech's position coefficients are single bytes in the EEPROM region, and
//! the vendor ships a non-zero D. Both facts change how `set_gain` has to work; see there.
//!
//! Written against `rustypot`'s `sts3215` definition, whose register addresses and encodings
//! are the ones used here rather than a transcription of the datasheet — a servo family with
//! a second-hand memory table is exactly where a transcription error looks right and walks
//! wrong.

use std::time::Duration;

use rustypot::servo::feetech::sts3215::Sts3215Controller;

use crate::imu::{IMU_BLOCK_LEN, STALE_RUN_WARN, SflpDecoder, StaleImuTracker};
use crate::io::{ImuStale, IoError, JointTargets, Result, RobotIo, Sensors, SlowSensors};
use crate::model::{
    BAUD_RATE, FT_ENDIAN_ADDR, FT_ENDIAN_LITTLE, FT_EXPECTED_REGISTERS, FT_FACTORY_ID,
    FT_GOAL_POSITION_ADDR, FT_ID_ADDR, FT_LOCK_ADDR, FT_MA_PER_CURRENT_UNIT, FT_MODE_ADDR,
    FT_MODE_POSITION, FT_OFF_CURRENT, FT_OFF_POSITION, FT_OFF_SPEED, FT_OFF_TEMPERATURE,
    FT_OFF_VOLTAGE, FT_P_GAIN_ADDR, FT_PHASE_ADDR, FT_PHASE_SPEED_UNIT_BIT, FT_RAD_PER_COUNT,
    FT_RAD_PER_COUNT_X50, FT_READ_ADDR, FT_READ_LEN, FT_TORQUE_ENABLE_ADDR, FT_VOLTS_PER_COUNT,
    IMU_DXL_ID, JOINT_IDS, JOINT_NAMES, NUM_JOINTS, feetech_position_counts, feetech_position_rad,
    sign_magnitude,
};

/// A healthy sixteen-device read completes well inside this. Same reasoning as the Dynamixel
/// bus: a missing device should cost a bounded hiccup, not stall the loop on the serial
/// driver's default.
const READ_TIMEOUT: Duration = Duration::from_millis(30);

/// Pause after each EEPROM write. The servo acknowledges before the cell is necessarily
/// committed, and the writes here happen once per motor swap.
const EEPROM_SETTLE: Duration = Duration::from_millis(20);

/// How long an HD-1910 is off the bus after a `0x08` before it answers again.
///
/// The datasheet says about 800 ms. The margin is not politeness: pinging a servo that is
/// merely still booting reads it as one whose flash failed, and would fail an adoption that
/// worked. This is only ever waited through by [`FeetechIo::adopt_replacement`], never by the
/// control loop — see [`RobotIo::reboot`].
const REBOOT_SETTLE: Duration = Duration::from_millis(1200);

/// `torque_enable` has no boolean register on this family the way it does on the XL330.
const TORQUE_OFF: u8 = 0;
const TORQUE_ON: u8 = 1;

/// `lock`. Write 1 to lock, 0 to unlock. A locked write to an EEPROM address is accepted and
/// not persisted — which is the documented behaviour `set_gain` relies on.
const FT_UNLOCKED: u8 = 0;
const FT_LOCKED: u8 = 1;

/// How the present-speed register is scaled, when a bench has measured something other than
/// what the servo reports about itself.
///
/// The register means one of two things depending on a bit in `phase`, and the two differ by
/// fifty. A wrong choice scales every joint velocity in the observation vector by fifty,
/// which a policy tolerates well enough to walk badly — so the default reads the bit, and
/// this exists for the case where the bit turns out to lie. `bus.speed_unit` in
/// `robotd.toml` is what selects it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpeedUnit {
    /// Believe the servo: `phase` BIT2 says one count is 1 step/s or 50 steps/s.
    #[default]
    Auto,
    /// One count is one step per second — the datasheet's 0.0146 RPM unit.
    Step,
    /// One count is fifty steps per second — the 0.732 RPM unit.
    Step50,
}

pub struct FeetechIo {
    controller: Sts3215Controller,
    /// IMU first, then the servos in [`JOINT_IDS`] order — the order blocks come back in.
    ids: Vec<u8>,
    /// Per joint, `+1.0` or `-1.0`: the servo's counting sense relative to the joint's. The
    /// one hardware fact this backend cannot discover for itself.
    directions: [f64; NUM_JOINTS],
    /// Turns the `gain` this robot is configured with into the P coefficient these servos
    /// take. Not one, because the two numbers do not mean the same thing; see
    /// [`RobotIo::set_gain`].
    p_scale: f64,
    /// What the configured [`SpeedUnit`] asked for, before `phase` is read.
    speed_unit: SpeedUnit,
    /// rad/s per present-speed count, resolved at startup from `phase` or from `speed_unit`.
    rad_per_sec_per_count: f64,
    imu: SflpDecoder,
    /// Blocks identical to their predecessor — a board answering without refreshing.
    stale_imu: StaleImuTracker,
    /// The last tick's voltage and thermals. They ride in the block the tick already reads,
    /// so there is nothing to poll; see [`RobotIo::slow_sensors`].
    slow: Option<SlowSensors>,
}

impl FeetechIo {
    /// Open the bus.
    ///
    /// `directions` is `bus.directions` from `robotd.toml` and `p_scale` is `bus.p_gain_scale`;
    /// `speed_unit` is `bus.speed_unit`. All three are hardware facts about *this* robot that
    /// the code cannot read off the wire, which is why they are settings rather than guesses.
    pub fn open(
        port: &str,
        directions: [f64; NUM_JOINTS],
        p_scale: f64,
        speed_unit: SpeedUnit,
    ) -> Result<Self> {
        let controller = open_controller(port, BAUD_RATE)?;

        let mut ids = Vec::with_capacity(NUM_JOINTS + 1);
        ids.push(IMU_DXL_ID);
        ids.extend_from_slice(&JOINT_IDS);

        Ok(Self {
            controller,
            ids,
            directions,
            p_scale,
            speed_unit,
            rad_per_sec_per_count: FT_RAD_PER_COUNT,
            imu: SflpDecoder::default(),
            stale_imu: StaleImuTracker::default(),
            slow: None,
        })
    }

    /// Verify — and, in the one case it is safe, correct — the registers the control loop
    /// depends on. Returns how many were corrected.
    ///
    /// Nothing here is cosmetic. A servo not at 1 Mbps could not be talking to this port at
    /// all; one at response level 0 answers reads and pings and nothing else, so every write
    /// the robot makes for the rest of its life would time out.
    ///
    /// The mode is read and *reported*, never written. Putting a servo in the wrong mode makes
    /// it ignore position commands, and writing the mode across fifteen servos at once is how
    /// a robot arrives gaitless with no way back — a person looks at that, not this code.
    pub fn check_registers(&mut self) -> Result<usize> {
        self.check_endianness()?;
        self.resolve_speed_unit()?;

        let mut fixed = 0;
        for &id in &JOINT_IDS {
            fixed += self.check_registers_of(id)?;
        }
        Ok(fixed)
    }

    /// [`Self::check_registers`] for one servo.
    fn check_registers_of(&mut self, id: u8) -> Result<usize> {
        let mut fixed = 0;
        for &(name, addr, want) in FT_EXPECTED_REGISTERS {
            let got = self.read_u8(id, addr, name)?;
            if got == want {
                continue;
            }
            // At response level 0 the servo does not acknowledge the write that would raise
            // it — so the write may land and there is no way to tell from here. Refused
            // rather than attempted, because "accepted, unknown whether applied" is a worse
            // state to leave a bus in than "nothing was sent and the message says why".
            if name == "response_status_level" {
                return Err(IoError::Bus(format!(
                    "servo {id} reports {name}={got}, needs {want}: at level 0 a servo \
                     acknowledges nothing but reads and pings, so no write reaches it. Set it \
                     with the vendor tool and restart."
                )));
            }
            tracing::warn!(id, register = name, got, want, "correcting motor register");
            self.write_u8(id, addr, want, name)?;
            std::thread::sleep(EEPROM_SETTLE);
            fixed += 1;
        }

        let mode = self.read_u8(id, FT_MODE_ADDR, "mode")?;
        if mode != FT_MODE_POSITION {
            tracing::warn!(
                id,
                mode,
                want = FT_MODE_POSITION,
                "servo is not in the pure-position mode this robot runs; reported, not changed"
            );
        }
        Ok(fixed)
    }

    /// `END` at address 2 says how multi-byte fields are ordered. Asserted once, off the first
    /// servo: every field this backend decodes is assembled little-endian, and a device that
    /// disagreed would hand the loop angles built from swapped bytes — plausible numbers from
    /// the wrong registers, which is the hardest kind of fault to see.
    fn check_endianness(&mut self) -> Result<()> {
        let id = JOINT_IDS[0];
        let endian = self.read_u8(id, FT_ENDIAN_ADDR, "END")?;
        if endian != FT_ENDIAN_LITTLE {
            return Err(IoError::Bus(format!(
                "servo {id} reports END={endian} at address {FT_ENDIAN_ADDR}; this bus decodes \
                 little-endian (END=0) and cannot read a big-endian control table"
            )));
        }
        Ok(())
    }

    /// Resolve the present-speed scale, from `phase` unless the configuration already said.
    ///
    /// Logged at `warn` deliberately. A wrong scale here fails nothing — the loop runs, the
    /// robot walks, the health gate passes — it just walks badly, and this line is the only
    /// place the number behind that is written down.
    fn resolve_speed_unit(&mut self) -> Result<()> {
        let id = JOINT_IDS[0];
        let phase = self.read_u8(id, FT_PHASE_ADDR, "phase")?;
        let from_bit = if phase & (1 << FT_PHASE_SPEED_UNIT_BIT) != 0 {
            FT_RAD_PER_COUNT_X50
        } else {
            FT_RAD_PER_COUNT
        };
        self.rad_per_sec_per_count = match self.speed_unit {
            SpeedUnit::Auto => from_bit,
            SpeedUnit::Step => FT_RAD_PER_COUNT,
            SpeedUnit::Step50 => FT_RAD_PER_COUNT_X50,
        };
        tracing::warn!(
            id,
            phase,
            from_phase_bit = from_bit,
            used = self.rad_per_sec_per_count,
            "feetech phase register read; this is the present-speed unit"
        );
        Ok(())
    }

    /// The expected servo IDs that do not answer a ping, in [`JOINT_IDS`] order.
    ///
    /// Fifteen pings, each bounded by [`READ_TIMEOUT`]. Run once at startup, by the same
    /// caller that runs it on a Dynamixel bus — what decides whether there is a servo to
    /// adopt, and the whole of what a complete bus pays for the swap path.
    pub fn missing_servos(&mut self) -> Result<Vec<u8>> {
        let mut missing = Vec::new();
        for &id in &JOINT_IDS {
            let answered = self
                .controller
                .ping(id)
                .map_err(|e| IoError::Bus(format!("ping {id}: {e}")))?;
            if !answered {
                missing.push(id);
            }
        }
        Ok(missing)
    }

    /// Flash a factory-fresh servo so it takes the place of the one that is missing.
    ///
    /// A new HD-1910 answers as ID 1 at 1 Mbps. The ID is the same number the XL330 ships
    /// with, so the census that finds it is shared; the *speed* is where this path is
    /// simpler, because 1 Mbps is what this bus already runs at — there is no port to reopen
    /// and no rate register to write.
    ///
    /// What it costs instead is the write lock. The factory ships `lock = 1`, and a locked
    /// write to an EEPROM address is accepted and **not persisted**; the id lives at address
    /// 5, inside that region. So the ID is written with the servo unlocked and the lock put
    /// back afterwards. Getting this wrong looks like it worked — the servo answers to its
    /// new number until the next power cycle, and then answers to nobody's.
    ///
    /// The reboot at the end is not for tidiness: it is what proves the ID survived a reset.
    /// A servo that comes back silent is a servo whose EEPROM write was dropped.
    ///
    /// Returns `Ok(false)` when nothing answers at the factory ID — the servo is simply
    /// missing, or was replaced by one that is not fresh. The bus is open at [`BAUD_RATE`]
    /// either way, so the caller can keep waiting on it.
    pub fn adopt_replacement(&mut self, id: u8) -> Result<bool> {
        let name = JOINT_IDS
            .iter()
            .position(|&j| j == id)
            .map(|i| JOINT_NAMES[i])
            .ok_or_else(|| IoError::Bus(format!("{id} is not a joint id")))?;

        let found = self
            .controller
            .ping(FT_FACTORY_ID)
            .map_err(|e| IoError::Bus(format!("ping factory id {FT_FACTORY_ID}: {e}")))?;
        if !found {
            return Ok(false);
        }
        tracing::warn!(
            id,
            joint = name,
            "factory-fresh servo on the bus; flashing it as the missing joint"
        );

        self.write_u8(FT_FACTORY_ID, FT_LOCK_ADDR, FT_UNLOCKED, "lock")?;
        std::thread::sleep(EEPROM_SETTLE);
        self.write_u8(FT_FACTORY_ID, FT_ID_ADDR, id, "id")?;
        std::thread::sleep(EEPROM_SETTLE);
        // Answered at the new number from here on, so the lock goes back on the id just
        // written rather than on the one it arrived with.
        self.write_u8(id, FT_LOCK_ADDR, FT_LOCKED, "lock")?;
        std::thread::sleep(EEPROM_SETTLE);

        // Now an ordinary servo at the right address: the same check the others get pins the
        // response level and reports the mode.
        let fixed = self.check_registers_of(id)?;

        self.reboot(id)?;
        std::thread::sleep(REBOOT_SETTLE);
        let back = self
            .controller
            .ping(id)
            .map_err(|e| IoError::Bus(format!("ping {id} after reboot: {e}")))?;
        if !back {
            return Err(IoError::Bus(format!(
                "servo {id} ({name}) was flashed but did not come back from its reboot — the id \
                 write did not persist, which is what a locked EEPROM does to it"
            )));
        }
        tracing::warn!(
            id,
            joint = name,
            registers_fixed = fixed,
            "replacement servo adopted"
        );
        Ok(true)
    }

    /// Present positions only — a lighter read than [`RobotIo::read`], used once at startup
    /// to adopt the pose the robot is already in.
    pub fn present_positions(&mut self) -> Result<[f64; NUM_JOINTS]> {
        let blocks = self
            .controller
            .sync_read_raw_data(&JOINT_IDS, FT_READ_ADDR, 2)
            .map_err(|e| IoError::Bus(format!("read present positions: {e}")))?;
        if blocks.len() != NUM_JOINTS {
            return Err(IoError::ShortRead {
                what: "present positions",
                expected: NUM_JOINTS,
                got: blocks.len(),
            });
        }
        let mut out = [0.0; NUM_JOINTS];
        for (joint, block) in blocks.iter().enumerate() {
            if block.len() != 2 {
                return Err(IoError::ShortRead {
                    what: "present position block",
                    expected: 2,
                    got: block.len(),
                });
            }
            let raw = u16::from_le_bytes([block[0], block[1]]);
            out[joint] = self.directions[joint] * feetech_position_rad(raw);
        }
        Ok(out)
    }

    /// Torque on every servo.
    ///
    /// One transaction per joint, so not something to call per tick — the control loop calls
    /// it once, when someone enables the policy on a limp robot.
    ///
    /// **Every servo is written, whatever the others said.** For the same reason as the
    /// Dynamixel backend: an error part-way through used to end the loop there, and on the
    /// way to a power-off that is a robot sitting down with half its legs still locked.
    pub fn set_torque(&mut self, on: bool) -> Result<()> {
        let value = if on { TORQUE_ON } else { TORQUE_OFF };
        let mut failed = Vec::new();
        for &id in &JOINT_IDS {
            if let Err(e) = self.write_u8(id, FT_TORQUE_ENABLE_ADDR, value, "torque_enable") {
                failed.push(e.to_string());
            }
        }
        if failed.is_empty() {
            Ok(())
        } else {
            Err(IoError::Bus(failed.join("; ")))
        }
    }

    /// Ramp every joint from where it is now to `target`, linearly.
    ///
    /// Only ever called by an explicit `init` — the control loop must never move the robot on
    /// its own, because that would make an update restart a fall risk. Blocking, deliberately:
    /// nothing else should be talking to the bus while this runs.
    pub fn interpolate_to(
        &mut self,
        target: &[f64; NUM_JOINTS],
        duration: Duration,
        step: Duration,
    ) -> Result<()> {
        let start = self.present_positions()?;
        let steps = (duration.as_secs_f64() / step.as_secs_f64())
            .ceil()
            .max(1.0) as u32;
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            let mut next = [0.0; NUM_JOINTS];
            for j in 0..NUM_JOINTS {
                next[j] = start[j] + (target[j] - start[j]) * t;
            }
            self.write(&JointTargets::new(next))?;
            std::thread::sleep(step);
        }
        Ok(())
    }

    /// Read one byte-wide register.
    fn read_u8(&mut self, id: u8, addr: u8, what: &str) -> Result<u8> {
        let raw = self
            .controller
            .read_raw_data(id, addr, 1)
            .map_err(|e| IoError::Bus(format!("read {what} on {id}: {e}")))?;
        raw.first().copied().ok_or(IoError::ShortRead {
            what: "register read",
            expected: 1,
            got: raw.len(),
        })
    }

    /// Write one byte-wide register.
    fn write_u8(&mut self, id: u8, addr: u8, value: u8, what: &str) -> Result<()> {
        self.controller
            .write_raw_data(id, addr, vec![value])
            .map_err(|e| IoError::Bus(format!("write {what}={value} on {id}: {e}")))
    }
}

/// One servo's fifteen bytes as the control loop wants them:
/// `(position rad, velocity rad/s, current mA, volts, °C)`.
///
/// Free rather than a method so it can be tested without a serial port. The offsets are the
/// whole of it: a wrong one hands a joint its neighbour's value, which reads as a wiring
/// fault rather than a bug, and this is the only place they are applied.
fn parse_motor_block(
    block: &[u8],
    direction: f64,
    rad_per_sec_per_count: f64,
) -> Result<(f64, f64, f64, f64, f64)> {
    if block.len() != FT_READ_LEN as usize {
        return Err(IoError::ShortRead {
            what: "motor block",
            expected: FT_READ_LEN as usize,
            got: block.len(),
        });
    }

    let position = u16::from_le_bytes([block[FT_OFF_POSITION], block[FT_OFF_POSITION + 1]]);
    let speed = u16::from_le_bytes([block[FT_OFF_SPEED], block[FT_OFF_SPEED + 1]]);
    let current = u16::from_le_bytes([block[FT_OFF_CURRENT], block[FT_OFF_CURRENT + 1]]);

    // The direction is applied to velocity as well as position, and that is not symmetry for
    // its own sake: a joint whose servo counts the other way runs backwards for a positive
    // command, so its velocity is negated by the same sign. Flipping one and not the other
    // feeds the policy a robot whose commands and feedback disagree about which way is
    // forward, which is worse than a scale error — it is silent.
    Ok((
        direction * feetech_position_rad(position),
        direction * sign_magnitude(speed) as f64 * rad_per_sec_per_count,
        (sign_magnitude(current) as f64 * FT_MA_PER_CURRENT_UNIT).abs(),
        block[FT_OFF_VOLTAGE] as f64 * FT_VOLTS_PER_COUNT,
        block[FT_OFF_TEMPERATURE] as f64,
    ))
}

/// One joint's goal, as the two bytes the bus takes.
fn encode_goal(rad: f64, direction: f64) -> [u8; 2] {
    feetech_position_counts(direction * rad).to_le_bytes()
}

/// The serial port at `baud`, wrapped in a Protocol 1 controller.
///
/// Protocol 1 framing is what Feetech speaks — `FF FF ID LEN INSTR … CHK`. There is no v2
/// here and no fast sync read: 0x8A is a Dynamixel instruction this family does not
/// implement, so a plain `sync_read` (0x82) is the only one there is, and the
/// `bus.fast_sync_read` setting has nothing to turn on.
fn open_controller(port: &str, baud: u32) -> Result<Sts3215Controller> {
    let serial = serialport::new(port, baud)
        .timeout(READ_TIMEOUT)
        .open()
        .map_err(|e| IoError::Port {
            path: port.to_owned(),
            source: std::io::Error::other(e),
        })?;
    Ok(Sts3215Controller::new()
        .with_protocol_v1()
        .with_serial_port(serial))
}

impl RobotIo for FeetechIo {
    fn read(&mut self) -> Result<Sensors> {
        let blocks = self
            .controller
            .sync_read_raw_data(&self.ids, FT_READ_ADDR, FT_READ_LEN)
            .map_err(|e| IoError::Bus(format!("combined imu+motor sync_read: {e}")))?;

        if blocks.len() != self.ids.len() {
            return Err(IoError::ShortRead {
                what: "sync_read blocks",
                expected: self.ids.len(),
                got: blocks.len(),
            });
        }

        let mut sensors = Sensors::default();

        // Slot 0 is the IMU board. Its twelve bytes sit at the same offset, in the same
        // encoding, as they do on a Dynamixel bus — the board answers at the same address
        // with its own fifteen-byte block, three bytes longer than the twelve the decoder
        // wants.
        //
        // The two bytes past the window are a sample counter and a status byte, and they are
        // outside it *on purpose*: a frozen board is told from a live one by comparing these
        // twelve bytes, so a board refreshing its counter without refreshing its orientation
        // has to keep reading as frozen. Decoding by counter would hide exactly the fault the
        // counter exists to expose.
        if blocks[0].len() != FT_READ_LEN as usize {
            return Err(IoError::ShortRead {
                what: "imu block",
                expected: FT_READ_LEN as usize,
                got: blocks[0].len(),
            });
        }
        let mut raw = [0u8; IMU_BLOCK_LEN];
        raw.copy_from_slice(&blocks[0][..IMU_BLOCK_LEN]);
        let run = self.stale_imu.observe(&raw);
        if run == STALE_RUN_WARN || (run > STALE_RUN_WARN && run.is_multiple_of(500)) {
            tracing::warn!(
                consecutive = run,
                total = self.stale_imu.stale().total,
                "imu board has returned the same sample {run} reads running — orientation is frozen"
            );
        }
        sensors.imu = self.imu.decode(&raw);

        let mut volts = Vec::with_capacity(NUM_JOINTS);
        let mut temps_c = [0.0; NUM_JOINTS];

        for (joint, block) in blocks[1..].iter().enumerate() {
            let (position, velocity, current_ma, volts_here, temp_c) =
                parse_motor_block(block, self.directions[joint], self.rad_per_sec_per_count)?;
            sensors.positions[joint] = position;
            sensors.velocities[joint] = velocity;
            sensors.currents_ma[joint] = current_ma;
            temps_c[joint] = temp_c;
            // The zero filter guards a device answering with a nonsense value, which must not
            // be averaged in as if the pack were half flat. Same rule as the Dynamixel bus.
            if volts_here > 0.0 {
                volts.push(volts_here);
            }
        }

        if !volts.is_empty() {
            // Voltage is averaged because all fifteen servos sit on one pack: a single
            // reading is the same measurement with more noise. Temperature is *not* averaged
            // — one loaded joint running hot is the case worth seeing, and a mean over
            // fifteen hides it.
            self.slow = Some(SlowSensors {
                volts: volts.iter().sum::<f64>() / volts.len() as f64,
                temps_c,
            });
        }

        Ok(sensors)
    }

    fn write(&mut self, targets: &JointTargets) -> Result<()> {
        // Two little-endian bytes per joint, already in the servo's own sense — the direction
        // is applied on the way out as well as the way in, or a robot whose servos count the
        // other way would command its home pose and get a mirror of it.
        let payload: Vec<Vec<u8>> = (0..NUM_JOINTS)
            .map(|joint| encode_goal(targets.positions[joint], self.directions[joint]).to_vec())
            .collect();
        self.controller
            .sync_write_raw_data(&JOINT_IDS, FT_GOAL_POSITION_ADDR, &payload)
            .map_err(|e| IoError::Bus(format!("sync_write goal positions: {e}")))
    }

    fn set_torque(&mut self, on: bool) -> Result<()> {
        // The inherent method, which `robotd init` uses.
        FeetechIo::set_torque(self, on)
    }

    /// Position P on every joint.
    ///
    /// Two things are deliberately different from the Dynamixel backend, and both are
    /// properties of the servo rather than preferences:
    ///
    /// **`kp` is scaled rather than copied.** It is this robot's own number — 200 running,
    /// 160 standing, 50 limp — and its meaning is the XL330's position P. A Feetech
    /// coefficient is a single byte, 0–254, whose unit is the vendor's own; the two are not
    /// the same quantity, so `bus.p_gain_scale` converts between them. A value that has not
    /// been calibrated against a bench step response is a starting point, not a tuning.
    ///
    /// **D and I are left alone.** The Dynamixel path pins them at zero because the XL330
    /// ships at zero, so "write zero" and "write nothing" are the same act there. An HD-1910
    /// ships with D=32, which is part of the vendor's tuning for a 320:1 gear train, and
    /// zeroing it is how a joint starts to oscillate under load.
    ///
    /// The write goes out with the EEPROM lock **in place**. That is the documented way to
    /// say "accept this and do not persist it", which gives the register the RAM semantics the
    /// XL330's coefficient has and is why a state change every few ticks does not wear the
    /// flash. If a bench ever finds that a locked write is dropped rather than deferred, this
    /// is the line to revisit — and `bus.p_gain_scale` is the escape hatch that turns the
    /// whole scheme into a torque limit instead.
    fn set_gain(&mut self, kp: u16) -> Result<()> {
        let p = (f64::from(kp) * self.p_scale).round().clamp(1.0, 254.0) as u8;
        for &id in &JOINT_IDS {
            self.write_u8(id, FT_P_GAIN_ADDR, p, "p_coefficient")?;
        }
        Ok(())
    }

    /// Reboot one servo: instruction `0x08`.
    ///
    /// The way out of a latched overload or overheat fault, which otherwise holds torque off
    /// until the battery is pulled. The servo is off the bus for the better part of a second
    /// and comes back with torque off and its RAM registers at their EEPROM values; the
    /// caller owns putting them back.
    ///
    /// The torque is dropped *before* the instruction, as the datasheet requires — a servo
    /// that restarts under load is the one case it does not survive.
    ///
    /// **Nothing waits here.** `0x08` has no reply, so a servo that says nothing is not an
    /// error, and a reboot takes about 800 ms — waiting for one joint inside the control loop
    /// would stall the tick for a second per servo. The waiting lives in
    /// [`FeetechIo::adopt_replacement`], which is the only caller that needs to know the
    /// servo came back.
    fn reboot(&mut self, id: u8) -> Result<()> {
        self.write_u8(id, FT_TORQUE_ENABLE_ADDR, TORQUE_OFF, "torque_enable")?;
        self.controller
            .reboot(id)
            .map(|_| ())
            .map_err(|e| IoError::Bus(format!("reboot {id}: {e}")))
    }

    /// Supply voltage and case temperatures — from the tick's own block, not a new
    /// transaction.
    ///
    /// This is the one place the Feetech register map is friendlier than the Dynamixel one:
    /// both live inside the fifteen bytes [`RobotIo::read`] already fetches, so there is no
    /// second transaction and no decision about how often to spend one. What the caller gets
    /// is the last tick's answer, which for a pack and a case is a second old at most.
    fn slow_sensors(&mut self) -> Result<SlowSensors> {
        self.slow.ok_or_else(|| {
            IoError::Bus("no tick has been read yet; no voltage or temperature to report".into())
        })
    }

    fn imu_stale(&self) -> ImuStale {
        self.stale_imu.stale()
    }

    fn imu_ready(&self) -> bool {
        self.imu.ready()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Only this module fills the load field — the parser above does not read it, and naming it
    // there would be an unused import in the daemon's build.
    use crate::model::FT_OFF_LOAD;

    /// A block with recognisable values in every field the parser touches.
    fn block() -> Vec<u8> {
        let mut b = vec![0u8; FT_READ_LEN as usize];
        b[FT_OFF_POSITION..FT_OFF_POSITION + 2].copy_from_slice(&2048u16.to_le_bytes());
        b[FT_OFF_SPEED..FT_OFF_SPEED + 2].copy_from_slice(&100u16.to_le_bytes());
        b[FT_OFF_LOAD..FT_OFF_LOAD + 2].copy_from_slice(&7u16.to_le_bytes());
        b[FT_OFF_VOLTAGE] = 74;
        b[FT_OFF_TEMPERATURE] = 32;
        b[FT_OFF_CURRENT..FT_OFF_CURRENT + 2].copy_from_slice(&200u16.to_le_bytes());
        b
    }

    /// Every field out of its own offset. A wrong one hands a joint its neighbour's value,
    /// which reads as a wiring fault rather than a bug — the reason the offsets are named
    /// constants and this test exists.
    #[test]
    fn a_motor_block_decodes_from_its_own_offsets() {
        let (position, velocity, current_ma, volts, temp_c) =
            parse_motor_block(&block(), 1.0, FT_RAD_PER_COUNT).unwrap();
        assert!(
            (position - 0.0).abs() < 1e-12,
            "centre should be 0 rad, got {position}"
        );
        assert!((velocity - 100.0 * FT_RAD_PER_COUNT).abs() < 1e-12);
        assert!((current_ma - 200.0 * FT_MA_PER_CURRENT_UNIT).abs() < 1e-12);
        assert!((volts - 7.4).abs() < 1e-12);
        assert!((temp_c - 32.0).abs() < 1e-12);
    }

    /// The direction flips the commanded sense, so it has to flip the feedback sense too —
    /// position *and* velocity. Flipping one and not the other is worse than a scale error:
    /// the robot's commands and its feedback would disagree about which way is forward, with
    /// nothing to show for it.
    #[test]
    fn the_direction_mirrors_position_and_velocity_together() {
        let mut b = block();
        b[FT_OFF_POSITION..FT_OFF_POSITION + 2].copy_from_slice(&3072u16.to_le_bytes());
        b[FT_OFF_SPEED..FT_OFF_SPEED + 2].copy_from_slice(&100u16.to_le_bytes());

        let (p_pos, v_pos, ..) = parse_motor_block(&b, 1.0, FT_RAD_PER_COUNT).unwrap();
        let (p_neg, v_neg, ..) = parse_motor_block(&b, -1.0, FT_RAD_PER_COUNT).unwrap();

        assert!((p_neg + p_pos).abs() < 1e-12, "position must mirror");
        assert!((v_neg + v_pos).abs() < 1e-12, "velocity must mirror");
    }

    /// Current is reported as load, so the sign is dropped — and dropped on the *magnitude*,
    /// not on the two's-complement reading of a sign-magnitude word. Reading 0x8001 as `i16`
    /// gives −32767, and a joint pushing hard would report as one pulling.
    #[test]
    fn current_is_reported_as_load_regardless_of_direction() {
        let mut b = block();
        b[FT_OFF_CURRENT..FT_OFF_CURRENT + 2].copy_from_slice(&0x8002u16.to_le_bytes());
        let (.., current_ma, _, _) = parse_motor_block(&b, 1.0, FT_RAD_PER_COUNT).unwrap();
        assert!((current_ma - 2.0 * FT_MA_PER_CURRENT_UNIT).abs() < 1e-12);
    }

    /// A short block means a device did not answer. Reported rather than papered over: the
    /// rest of the array would keep stale values, and a half-updated sample is what the
    /// safety layer reads.
    #[test]
    fn a_block_of_the_wrong_length_is_a_short_read() {
        assert!(parse_motor_block(&block()[..12], 1.0, FT_RAD_PER_COUNT).is_err());
        assert!(parse_motor_block(&[0u8; 20], 1.0, FT_RAD_PER_COUNT).is_err());
    }

    /// The two speed units differ by fifty, which is the whole reason `bus.speed_unit` exists:
    /// measuring the wrong one scales every joint velocity the policy sees.
    #[test]
    fn the_speed_scale_is_what_selects_the_unit() {
        let (_, one_step, ..) = parse_motor_block(&block(), 1.0, FT_RAD_PER_COUNT).unwrap();
        let (_, fifty_step, ..) = parse_motor_block(&block(), 1.0, FT_RAD_PER_COUNT_X50).unwrap();
        assert!((fifty_step / one_step - 50.0).abs() < 1e-12);
    }

    /// Goals go out little-endian, like everything else on this bus. Big-endian bytes put a
    /// joint at a plausible-looking but entirely different angle.
    #[test]
    fn a_goal_is_two_little_endian_bytes() {
        assert_eq!(encode_goal(0.0, 1.0), 2048u16.to_le_bytes());
        assert_eq!(
            encode_goal(std::f64::consts::FRAC_PI_2, 1.0),
            3072u16.to_le_bytes()
        );
    }

    /// The direction mirrors the goal the same way it mirrors the reading — otherwise a
    /// robot whose servos count the other way would command its home pose and get the
    /// mirror of it, standing crooked by twice every joint offset.
    #[test]
    fn the_direction_mirrors_the_goal_as_well() {
        let a = encode_goal(0.5, -1.0);
        let b = encode_goal(-0.5, 1.0);
        assert_eq!(a, b);
    }
}
