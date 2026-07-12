const scales = {
  major: { name: "Ionian major", intervals: [0, 2, 4, 5, 7, 9, 11], tonic: 60 },
  minor: { name: "Natural minor", intervals: [0, 2, 3, 5, 7, 8, 10], tonic: 57 },
  dorian: { name: "Dorian", intervals: [0, 2, 3, 5, 7, 9, 10], tonic: 62 },
  kalyani: { name: "Kalyani", intervals: [0, 2, 4, 6, 7, 9, 11], tonic: 60 },
  mayamalavagowla: { name: "Mayamalavagowla", intervals: [0, 1, 4, 5, 7, 8, 11], tonic: 60 },
  bhairav: { name: "Bhairav", intervals: [0, 1, 4, 5, 7, 8, 10], tonic: 60 }
};

const goalProfiles = {
  learnable: { leaps: 0.2, rests: 0.1, repeat: 0.44, density: 0.72 },
  novel: { leaps: 0.48, rests: 0.16, repeat: 0.18, density: 0.82 },
  cinematic: { leaps: 0.34, rests: 0.08, repeat: 0.3, density: 0.68 },
  groove: { leaps: 0.24, rests: 0.08, repeat: 0.28, density: 0.9 }
};

const form = document.querySelector("#controlForm");
const briefInput = document.querySelector("#briefInput");
const modeSelect = document.querySelector("#modeSelect");
const goalSelect = document.querySelector("#goalSelect");
const bpmInput = document.querySelector("#bpmInput");
const barsInput = document.querySelector("#barsInput");
const temperatureInput = document.querySelector("#temperatureInput");
const noveltyInput = document.querySelector("#noveltyInput");
const tutorToggle = document.querySelector("#tutorToggle");
const criticToggle = document.querySelector("#criticToggle");
const mutationToggle = document.querySelector("#mutationToggle");
const temperatureValue = document.querySelector("#temperatureValue");
const noveltyValue = document.querySelector("#noveltyValue");
const seedBadge = document.querySelector("#seedBadge");
const originalityScore = document.querySelector("#originalityScore");
const statusBadge = document.querySelector("#statusBadge");
const tokenStrip = document.querySelector("#tokenStrip");
const agentBoard = document.querySelector("#agentBoard");
const lineageList = document.querySelector("#lineageList");
const canvas = document.querySelector("#pianoRoll");
const ctx = canvas.getContext("2d");

let audioContext = null;
let activeNodes = [];
let currentMotif = null;
let lineage = [];
let generationCount = 0;

function hashString(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function mulberry32(seed) {
  return function random() {
    let t = seed += 0x6d2b79f5;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function pick(random, items) {
  return items[Math.floor(random() * items.length)];
}

function midiToFreq(note) {
  return 440 * Math.pow(2, (note - 69) / 12);
}

function degreeToMidi(scale, degree, octaveShift = 0) {
  const intervals = scale.intervals;
  const wrapped = ((degree % intervals.length) + intervals.length) % intervals.length;
  const octave = Math.floor(degree / intervals.length) + octaveShift;
  return scale.tonic + intervals[wrapped] + octave * 12;
}

function currentOptions(extraSalt = "") {
  const source = [
    briefInput.value,
    modeSelect.value,
    goalSelect.value,
    bpmInput.value,
    barsInput.value,
    temperatureInput.value,
    noveltyInput.value,
    extraSalt
  ].join("|");

  return {
    seed: hashString(source),
    brief: briefInput.value.trim(),
    scale: scales[modeSelect.value],
    modeKey: modeSelect.value,
    goal: goalSelect.value,
    bpm: clamp(Number(bpmInput.value) || 92, 52, 168),
    bars: clamp(Number(barsInput.value) || 4, 2, 8),
    temperature: Number(temperatureInput.value),
    novelty: Number(noveltyInput.value)
  };
}

function generateMotif(extraSalt = "") {
  const options = currentOptions(extraSalt);
  const random = mulberry32(options.seed);
  const profile = goalProfiles[options.goal];
  const steps = options.bars * 16;
  const phrase = [];
  const rhythmChoices = options.goal === "groove" ? [1, 1, 1, 2, 2, 3] : [1, 1, 2, 2, 3, 4];
  const chordDegrees = options.goal === "cinematic" ? [0, 5, 3, 4] : [0, 4, 5, 3];
  let cursor = 0;
  let degree = Math.floor(random() * 5);
  let lastDegree = degree;

  while (cursor < steps) {
    const duration = Math.min(pick(random, rhythmChoices), steps - cursor);
    const repeatChance = profile.repeat - options.temperature * 0.18;
    const leapChance = profile.leaps + options.temperature * 0.28 + options.novelty / 320;
    const shouldRest = random() > profile.density && cursor % 4 !== 0;

    if (random() < repeatChance) {
      degree = lastDegree;
    } else {
      const movement = random() < leapChance ? pick(random, [-4, -3, 3, 4, 5]) : pick(random, [-2, -1, 1, 2]);
      degree = clamp(lastDegree + movement, -5, 14);
    }

    const octaveShift = degree > 8 ? 0 : random() < 0.16 + options.temperature * 0.12 ? 1 : 0;
    const note = shouldRest ? null : degreeToMidi(options.scale, degree, octaveShift);
    const velocity = Math.round(58 + random() * 34 + (cursor % 8 === 0 ? 10 : 0));

    phrase.push({
      start: cursor,
      duration,
      degree,
      note,
      velocity,
      chord: chordDegrees[Math.floor(cursor / 16) % chordDegrees.length]
    });

    if (!shouldRest) {
      lastDegree = degree;
    }
    cursor += duration;
  }

  const motif = {
    id: `M${String(generationCount + 1).padStart(2, "0")}`,
    options,
    phrase,
    chords: chordDegrees,
    originality: scoreOriginality(phrase, options),
    contour: phrase.filter((event) => event.note !== null).map((event) => event.degree)
  };

  generationCount += 1;
  currentMotif = motif;
  lineage.unshift({
    id: motif.id,
    title: `${options.scale.name}, ${options.bpm} BPM`,
    detail: `${phrase.length} tokens, ${motif.originality}% originality`,
    score: motif.originality
  });
  lineage = lineage.slice(0, 7);
  renderMotif(motif);
}

function scoreOriginality(phrase, options) {
  const playable = phrase.filter((event) => event.note !== null);
  const uniqueDegrees = new Set(playable.map((event) => event.degree)).size;
  const intervalVariety = new Set(playable.slice(1).map((event, index) => event.degree - playable[index].degree)).size;
  const rhythmVariety = new Set(phrase.map((event) => event.duration)).size;
  const noveltyLift = Math.round(options.novelty * 0.22 + options.temperature * 16);
  return clamp(46 + uniqueDegrees * 3 + intervalVariety * 4 + rhythmVariety * 5 + noveltyLift, 51, 97);
}

function renderMotif(motif) {
  seedBadge.textContent = `Seed ${motif.options.seed % 10000}`;
  originalityScore.textContent = motif.originality;
  statusBadge.textContent = "Generated";
  drawPianoRoll(motif);
  renderTokens(motif);
  renderAgents(motif);
  renderLineage();
}

function drawPianoRoll(motif) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(640, Math.floor(rect.width * dpr));
  canvas.height = Math.max(300, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const width = canvas.width / dpr;
  const height = canvas.height / dpr;
  ctx.clearRect(0, 0, width, height);

  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#28231f");
  gradient.addColorStop(0.55, "#322820");
  gradient.addColorStop(1, "#1f302f");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  const padding = 28;
  const steps = motif.options.bars * 16;
  const minNote = 52;
  const maxNote = 82;
  const usableWidth = width - padding * 2;
  const usableHeight = height - padding * 2;

  ctx.strokeStyle = "rgba(255,255,255,0.09)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= steps; i += 4) {
    const x = padding + (i / steps) * usableWidth;
    ctx.beginPath();
    ctx.moveTo(x, padding);
    ctx.lineTo(x, height - padding);
    ctx.stroke();
  }

  for (let i = 0; i <= 7; i += 1) {
    const y = padding + (i / 7) * usableHeight;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(width - padding, y);
    ctx.stroke();
  }

  motif.chords.forEach((degree, index) => {
    const x = padding + (index / motif.chords.length) * usableWidth;
    const w = usableWidth / motif.chords.length;
    ctx.fillStyle = index % 2 === 0 ? "rgba(22,125,127,0.14)" : "rgba(216,92,67,0.11)";
    ctx.fillRect(x, padding, w, usableHeight);
    ctx.fillStyle = "rgba(255,255,255,0.62)";
    ctx.font = "700 12px system-ui";
    ctx.fillText(`deg ${degree + 1}`, x + 10, height - 12);
  });

  motif.phrase.forEach((event) => {
    if (event.note === null) return;
    const x = padding + (event.start / steps) * usableWidth;
    const w = Math.max(10, (event.duration / steps) * usableWidth - 3);
    const yRatio = (event.note - minNote) / (maxNote - minNote);
    const y = padding + (1 - yRatio) * usableHeight;
    const h = 13;
    ctx.fillStyle = event.duration > 2 ? "#f2b84b" : "#65d4b0";
    ctx.shadowColor = "rgba(101,212,176,0.28)";
    ctx.shadowBlur = 10;
    roundRect(ctx, x, y - h / 2, w, h, 5);
    ctx.fill();
    ctx.shadowBlur = 0;
  });
}

function roundRect(context, x, y, width, height, radius) {
  context.beginPath();
  context.moveTo(x + radius, y);
  context.arcTo(x + width, y, x + width, y + height, radius);
  context.arcTo(x + width, y + height, x, y + height, radius);
  context.arcTo(x, y + height, x, y, radius);
  context.arcTo(x, y, x + width, y, radius);
  context.closePath();
}

function renderTokens(motif) {
  const visible = motif.phrase.slice(0, 8);
  tokenStrip.innerHTML = visible.map((event) => {
    const label = event.note === null ? "REST" : `D${event.degree >= 0 ? "+" : ""}${event.degree}`;
    return `<div class="token"><strong>${label}</strong><span>DUR_${event.duration} VEL_${event.velocity}</span></div>`;
  }).join("");
}

function renderAgents(motif) {
  const playable = motif.phrase.filter((event) => event.note !== null);
  const first = playable[0]?.degree ?? 0;
  const last = playable[playable.length - 1]?.degree ?? 0;
  const leaps = playable.slice(1).filter((event, index) => Math.abs(event.degree - playable[index].degree) > 2).length;
  const cards = [
    {
      show: true,
      tone: "teal",
      name: "Composer Agent",
      text: `Built a ${motif.options.bars}-bar motif in ${motif.options.scale.name}; contour moves from degree ${first + 1} to degree ${last + 1}.`
    },
    {
      show: tutorToggle.checked,
      tone: "violet",
      name: "Theory Tutor",
      text: `Practice this as two call-and-response cells. Count the long notes first, then add passing tones at ${motif.options.bpm} BPM.`
    },
    {
      show: criticToggle.checked,
      tone: "coral",
      name: "Critic Agent",
      text: `${leaps} larger leaps, ${new Set(motif.phrase.map((event) => event.duration)).size} rhythm values, originality score ${motif.originality}.`
    },
    {
      show: mutationToggle.checked,
      tone: "mustard",
      name: "Mutation Agent",
      text: `Next variations: invert bars 2-3, reharmonize degree ${motif.chords[1] + 1}, or reduce the rhythm to quarter-note skeleton.`
    }
  ];

  agentBoard.innerHTML = cards.filter((card) => card.show).map((card) => `
    <article class="agent-card" data-tone="${card.tone}">
      <h3>${card.name}</h3>
      <p>${card.text}</p>
    </article>
  `).join("");
}

function renderLineage() {
  lineageList.innerHTML = lineage.map((item, index) => `
    <article class="lineage-item">
      <div class="node">${index + 1}</div>
      <div>
        <h3>${item.id} ${item.title}</h3>
        <p>${item.detail}</p>
      </div>
      <span>${item.score}%</span>
    </article>
  `).join("");
}

function ensureAudio() {
  if (!audioContext) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  return audioContext;
}

function stopPlayback() {
  activeNodes.forEach((node) => {
    try {
      node.stop();
    } catch (error) {
      return null;
    }
  });
  activeNodes = [];
  statusBadge.textContent = currentMotif ? "Stopped" : "Ready";
}

function playMotif() {
  if (!currentMotif) {
    generateMotif();
  }
  stopPlayback();
  const context = ensureAudio();
  if (context.state === "suspended") {
    context.resume();
  }
  const motif = currentMotif;
  const beat = 60 / motif.options.bpm;
  const step = beat / 4;
  const startTime = context.currentTime + 0.08;
  const master = context.createGain();
  master.gain.value = 0.42;
  master.connect(context.destination);

  motif.phrase.forEach((event) => {
    if (event.note === null) return;
    const osc = context.createOscillator();
    const gain = context.createGain();
    const filter = context.createBiquadFilter();
    const eventStart = startTime + event.start * step;
    const eventDuration = Math.max(0.08, event.duration * step * 0.9);

    osc.type = event.duration > 2 ? "triangle" : "sine";
    osc.frequency.value = midiToFreq(event.note);
    filter.type = "lowpass";
    filter.frequency.value = 1400 + event.velocity * 18;
    gain.gain.setValueAtTime(0.0001, eventStart);
    gain.gain.exponentialRampToValueAtTime((event.velocity / 127) * 0.34, eventStart + 0.018);
    gain.gain.exponentialRampToValueAtTime(0.0001, eventStart + eventDuration);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(master);
    osc.start(eventStart);
    osc.stop(eventStart + eventDuration + 0.03);
    activeNodes.push(osc);
  });

  motif.chords.forEach((degree, index) => {
    const root = degreeToMidi(motif.options.scale, degree, -1);
    [0, 2, 4].forEach((offset) => {
      const osc = context.createOscillator();
      const gain = context.createGain();
      const note = degreeToMidi(motif.options.scale, degree + offset, -1);
      const eventStart = startTime + index * 16 * step;
      const eventDuration = 15.6 * step;
      osc.type = "sawtooth";
      osc.frequency.value = midiToFreq(note || root);
      gain.gain.setValueAtTime(0.0001, eventStart);
      gain.gain.exponentialRampToValueAtTime(0.045, eventStart + 0.08);
      gain.gain.exponentialRampToValueAtTime(0.0001, eventStart + eventDuration);
      osc.connect(gain);
      gain.connect(master);
      osc.start(eventStart);
      osc.stop(eventStart + eventDuration + 0.04);
      activeNodes.push(osc);
    });
  });

  const total = motif.options.bars * 16 * step * 1000;
  statusBadge.textContent = "Playing";
  window.setTimeout(() => {
    if (statusBadge.textContent === "Playing") {
      statusBadge.textContent = "Ready";
    }
  }, total + 200);
}

temperatureInput.addEventListener("input", () => {
  temperatureValue.textContent = Number(temperatureInput.value).toFixed(2);
});

noveltyInput.addEventListener("input", () => {
  noveltyValue.textContent = noveltyInput.value;
});

document.querySelector("#generateBtn").addEventListener("click", () => generateMotif());
document.querySelector("#playBtn").addEventListener("click", playMotif);
document.querySelector("#stopBtn").addEventListener("click", stopPlayback);
document.querySelector("#mutateBtn").addEventListener("click", () => generateMotif(`mutation-${Date.now()}`));
form.addEventListener("change", () => {
  if (currentMotif) {
    generateMotif("control-change");
  }
});

window.addEventListener("resize", () => {
  if (currentMotif) {
    drawPianoRoll(currentMotif);
  }
});

generateMotif();
