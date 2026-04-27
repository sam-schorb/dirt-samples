#!/usr/bin/env node
'use strict';

/**
 * PC12 -> Throttler -> IAC bus (for Strudel)
 *
 * Goals:
 * - Coalesce high-rate CC streams (send only the latest value per CC per tick)
 * - Rate-limit outgoing CC updates (default 120 Hz)
 * - Pass through non-CC messages immediately (notes, pitchbend, etc.)
 * - Keep CPU usage low (O(changed CCs) per tick, not O(incoming events))
 *
 * Usage:
 *   node pc12-throttle.js --list
 *   node pc12-throttle.js --in "Faderfox PC12" --out "IAC Driver Bus 1"
 *   node pc12-throttle.js --in "Faderfox PC12" --out "Strudel Throttled" --hz 120
 */

const midi = require('@julusian/midi');

// -------------------- simple arg parsing --------------------
const argv = process.argv.slice(2);
const hasFlag = f => argv.includes(f);
const getArg = (name, def) => {
  const i = argv.findIndex(x => x === name);
  if (i === -1) return def;
  const v = argv[i + 1];
  return v === undefined || v.startsWith('--') ? def : v;
};

const LIST = hasFlag('--list');
const VERBOSE = hasFlag('--verbose') || hasFlag('--stats'); // stats on by default
const INPUT_WANT = getArg('--in', 'Faderfox PC12');
const OUTPUT_WANT = getArg('--out', 'IAC Driver Bus 1');

// Throttle parameters (tune here or via flags)
const FLUSH_HZ = Number(getArg('--hz', '120')); // 60–240 is typical
const DEAD_BAND = Number(getArg('--deadband', '1')); // 1 = drop duplicates only; 2+ reduces traffic more
const MAX_SEND_PER_TICK = Number(getArg('--max', '64')); // safety cap
const FLUSH_MS = Math.max(1, Math.floor(1000 / Math.max(1, FLUSH_HZ)));

// -------------------- port utilities --------------------
function listPorts() {
  const input = new midi.Input();
  const output = new midi.Output();

  console.log('\n[MIDI INPUT PORTS]');
  for (let i = 0; i < input.getPortCount(); i++) {
    console.log(`  ${i}: ${input.getPortName(i)}`);
  }

  console.log('\n[MIDI OUTPUT PORTS]');
  for (let i = 0; i < output.getPortCount(); i++) {
    console.log(`  ${i}: ${output.getPortName(i)}`);
  }
  console.log('');
}

function findPortIndex(kind, device, wantedName) {
  const want = (wantedName || '').toLowerCase().trim();
  const count = device.getPortCount();

  // Exact match first
  for (let i = 0; i < count; i++) {
    const name = device.getPortName(i);
    if (name.toLowerCase().trim() === want) return i;
  }
  // Substring match
  for (let i = 0; i < count; i++) {
    const name = device.getPortName(i);
    if (name.toLowerCase().includes(want)) return i;
  }

  // Not found
  return -1;
}

if (LIST) {
  listPorts();
  process.exit(0);
}

// -------------------- open ports --------------------
const input = new midi.Input();
const output = new midi.Output();

// Keep extra chatter down: sysex/timing/active-sensing are ignored by default in this lib.
// We leave defaults (good for performance). :contentReference[oaicite:2]{index=2}

const inIndex = findPortIndex('input', input, INPUT_WANT);
if (inIndex < 0) {
  console.error(
    `\n[ERROR] Could not find MIDI input containing: "${INPUT_WANT}"`
  );
  listPorts();
  process.exit(1);
}

const outIndex = findPortIndex('output', output, OUTPUT_WANT);
if (outIndex < 0) {
  console.error(
    `\n[ERROR] Could not find MIDI output containing: "${OUTPUT_WANT}"`
  );
  console.error(
    'Make sure your IAC Driver is online and the bus exists (Audio MIDI Setup).'
  );
  listPorts();
  process.exit(1);
}

input.openPort(inIndex);
output.openPort(outIndex);

const inName = input.getPortName(inIndex);
const outName = output.getPortName(outIndex);

console.log(`\n[OK] Throttling from input:  ${inName}`);
console.log(`[OK] Sending to output:     ${outName}`);
console.log(
  `[OK] CC flush: ${FLUSH_HZ} Hz (${FLUSH_MS} ms), deadband: ${DEAD_BAND}, max/tick: ${MAX_SEND_PER_TICK}`
);
console.log(
  '[TIP] In Strudel, use midin(...) on the IAC bus, not "Faderfox PC12".\n'
);

// -------------------- throttle core --------------------
// key: "ch:cc" -> latest message [status, cc, value]
const pending = new Map();
const changed = new Set();
const lastSentValue = new Map();

// stats
let inCount = 0;
let outCount = 0;
let droppedCount = 0;

function isCC(msg) {
  // status high nibble 0xB0..0xBF
  return msg && msg.length >= 3 && (msg[0] & 0xf0) === 0xb0;
}

input.on('message', (_deltaTime, msg) => {
  inCount++;

  if (isCC(msg)) {
    const ch = msg[0] & 0x0f;
    const cc = msg[1] & 0x7f;
    const val = msg[2] & 0x7f;
    const key = `${ch}:${cc}`;

    // coalesce: always keep only latest value for this CC until next flush tick
    pending.set(key, [msg[0], cc, val]);
    changed.add(key);
    return;
  }

  // pass-through for non-CC
  output.sendMessage(msg);
  outCount++;
});

setInterval(() => {
  let sentThisTick = 0;

  for (const key of changed) {
    if (sentThisTick >= MAX_SEND_PER_TICK) break;

    const msg = pending.get(key);
    if (!msg) {
      changed.delete(key);
      continue;
    }

    const val = msg[2];
    const last = lastSentValue.get(key);

    // DEAD_BAND=1 -> drop duplicates only; DEAD_BAND>1 -> ignore tiny changes
    const shouldSend =
      last === undefined ? true : Math.abs(val - last) >= DEAD_BAND;

    if (shouldSend) {
      output.sendMessage(msg);
      outCount++;
      lastSentValue.set(key, val);
    } else {
      droppedCount++;
    }

    changed.delete(key);
    sentThisTick++;
  }
}, FLUSH_MS);

if (VERBOSE) {
  setInterval(() => {
    const pendingCount = changed.size;
    console.log(
      `[stats] in=${inCount}/s out=${outCount}/s dropped=${droppedCount}/s pending=${pendingCount}`
    );
    inCount = 0;
    outCount = 0;
    droppedCount = 0;
  }, 1000).unref();
}

// clean shutdown
function shutdown() {
  console.log('\n[shutdown] closing MIDI ports...');
  try {
    input.closePort();
  } catch {}
  try {
    output.closePort();
  } catch {}
  process.exit(0);
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
