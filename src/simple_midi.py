"""Small Standard MIDI File reader/writer for symbolic-music experiments.

This module intentionally avoids external dependencies so the tokenizer can run
on a fresh Python installation. It supports the subset needed for melody-first
LLM training: note on/off, tempo, running status, and type 0/1 MIDI files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct


DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_TEMPO_US_PER_BEAT = 500_000


@dataclass(frozen=True)
class MidiNote:
    start: int
    end: int
    pitch: int
    velocity: int = 80
    channel: int = 0

    @property
    def duration(self) -> int:
        return max(0, self.end - self.start)


@dataclass(frozen=True)
class MidiSong:
    ticks_per_beat: int
    notes: list[MidiNote]
    tempo_us_per_beat: int = DEFAULT_TEMPO_US_PER_BEAT


def read_varlen(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, offset


def write_varlen(value: int) -> bytes:
    value = max(0, int(value))
    buffer = value & 0x7F
    value >>= 7
    while value:
        buffer <<= 8
        buffer |= (value & 0x7F) | 0x80
        value >>= 7

    out = bytearray()
    while True:
        out.append(buffer & 0xFF)
        if buffer & 0x80:
            buffer >>= 8
        else:
            break
    return bytes(out)


def read_midi(path: str | Path) -> MidiSong:
    raw = Path(path).read_bytes()
    offset = 0
    if raw[offset : offset + 4] != b"MThd":
        raise ValueError(f"{path} is not a Standard MIDI file")
    offset += 4
    header_length = struct.unpack(">I", raw[offset : offset + 4])[0]
    offset += 4
    fmt, track_count, ticks_per_beat = struct.unpack(">HHH", raw[offset : offset + 6])
    offset += header_length
    if fmt not in {0, 1}:
        raise ValueError(f"Unsupported MIDI format {fmt}; only type 0 and 1 are supported")

    notes: list[MidiNote] = []
    tempo = DEFAULT_TEMPO_US_PER_BEAT

    for _ in range(track_count):
        if raw[offset : offset + 4] != b"MTrk":
            raise ValueError("Malformed MIDI track chunk")
        offset += 4
        track_length = struct.unpack(">I", raw[offset : offset + 4])[0]
        offset += 4
        track = raw[offset : offset + track_length]
        offset += track_length

        track_notes, track_tempo = _parse_track(track)
        notes.extend(track_notes)
        if track_tempo is not None:
            tempo = track_tempo

    notes.sort(key=lambda note: (note.start, note.pitch, note.channel))
    return MidiSong(ticks_per_beat=ticks_per_beat, notes=notes, tempo_us_per_beat=tempo)


def _parse_track(track: bytes) -> tuple[list[MidiNote], int | None]:
    cursor = 0
    absolute = 0
    running_status: int | None = None
    active: dict[tuple[int, int], list[tuple[int, int]]] = {}
    notes: list[MidiNote] = []
    tempo: int | None = None

    while cursor < len(track):
        delta, cursor = read_varlen(track, cursor)
        absolute += delta
        status = track[cursor]

        if status < 0x80:
            if running_status is None:
                raise ValueError("MIDI running status used before a status byte")
            status = running_status
        else:
            cursor += 1
            if status < 0xF0:
                running_status = status

        if status == 0xFF:
            meta_type = track[cursor]
            cursor += 1
            length, cursor = read_varlen(track, cursor)
            payload = track[cursor : cursor + length]
            cursor += length
            if meta_type == 0x51 and length == 3:
                tempo = int.from_bytes(payload, "big")
            if meta_type == 0x2F:
                break
            continue

        if status in {0xF0, 0xF7}:
            length, cursor = read_varlen(track, cursor)
            cursor += length
            continue

        event_type = status & 0xF0
        channel = status & 0x0F
        data_len = 1 if event_type in {0xC0, 0xD0} else 2
        data = track[cursor : cursor + data_len]
        cursor += data_len

        if event_type == 0x90:
            pitch = data[0]
            velocity = data[1]
            key = (channel, pitch)
            if velocity == 0:
                _finish_note(active, notes, key, absolute)
            else:
                active.setdefault(key, []).append((absolute, velocity))
        elif event_type == 0x80:
            pitch = data[0]
            key = (channel, pitch)
            _finish_note(active, notes, key, absolute)

    for (channel, pitch), starts in active.items():
        for start, velocity in starts:
            if absolute > start:
                notes.append(MidiNote(start=start, end=absolute, pitch=pitch, velocity=velocity, channel=channel))

    return notes, tempo


def _finish_note(
    active: dict[tuple[int, int], list[tuple[int, int]]],
    notes: list[MidiNote],
    key: tuple[int, int],
    end: int,
) -> None:
    starts = active.get(key)
    if not starts:
        return
    start, velocity = starts.pop(0)
    if not starts:
        active.pop(key, None)
    if end > start:
        channel, pitch = key
        notes.append(MidiNote(start=start, end=end, pitch=pitch, velocity=velocity, channel=channel))


def write_midi(
    path: str | Path,
    notes: list[MidiNote],
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
    tempo_us_per_beat: int = DEFAULT_TEMPO_US_PER_BEAT,
) -> None:
    events: list[tuple[int, int, bytes]] = []
    events.append((0, 0, b"\xFF\x51\x03" + int(tempo_us_per_beat).to_bytes(3, "big")))
    events.append((0, 1, b"\xFF\x58\x04\x04\x02\x18\x08"))

    for note in notes:
        start = max(0, int(note.start))
        end = max(start + 1, int(note.end))
        pitch = max(0, min(127, int(note.pitch)))
        velocity = max(1, min(127, int(note.velocity)))
        channel = max(0, min(15, int(note.channel)))
        events.append((start, 2, bytes([0x90 | channel, pitch, velocity])))
        events.append((end, 1, bytes([0x80 | channel, pitch, 0])))

    events.sort(key=lambda event: (event[0], event[1]))
    track = bytearray()
    previous_tick = 0
    for tick, _, payload in events:
        track.extend(write_varlen(tick - previous_tick))
        track.extend(payload)
        previous_tick = tick
    track.extend(write_varlen(0))
    track.extend(b"\xFF\x2F\x00")

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticks_per_beat)
    chunk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(header + chunk)

