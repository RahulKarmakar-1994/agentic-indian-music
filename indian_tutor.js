const form = document.querySelector("#generateForm");
const generateButton = document.querySelector("#generateButton");
const statusBadge = document.querySelector("#statusBadge");
const deviceBadge = document.querySelector("#deviceBadge");
const scoreBadge = document.querySelector("#scoreBadge");
const ragaBadge = document.querySelector("#ragaBadge");
const validMetric = document.querySelector("#validMetric");
const notesMetric = document.querySelector("#notesMetric");
const swarasMetric = document.querySelector("#swarasMetric");
const secondsMetric = document.querySelector("#secondsMetric");
const swaraBoard = document.querySelector("#swaraBoard");
const audioPlayer = document.querySelector("#audioPlayer");
const lessonOutput = document.querySelector("#lessonOutput");
const tokenOutput = document.querySelector("#tokenOutput");
const tokenCount = document.querySelector("#tokenCount");
const midiLink = document.querySelector("#midiLink");
const temperatureInput = document.querySelector("#temperatureInput");
const temperatureValue = document.querySelector("#temperatureValue");
const lengthInput = document.querySelector("#lengthInput");
const lengthValue = document.querySelector("#lengthValue");

const swaraNames = {
  SWARA_SA: "Sa",
  SWARA_RE_KOMAL: "r",
  SWARA_RE: "Re",
  SWARA_GA_KOMAL: "g",
  SWARA_GA: "Ga",
  SWARA_MA: "Ma",
  SWARA_MA_TIVRA: "M^",
  SWARA_PA: "Pa",
  SWARA_DHA_KOMAL: "d",
  SWARA_DHA: "Dha",
  SWARA_NI_KOMAL: "n",
  SWARA_NI: "Ni"
};

function setStatus(text) {
  statusBadge.textContent = text;
}

function payloadFromForm() {
  return {
    raga: document.querySelector("#ragaInput").value,
    tala: document.querySelector("#talaInput").value,
    sa: document.querySelector("#saInput").value,
    tempo: Number(document.querySelector("#tempoInput").value),
    temperature: Number(temperatureInput.value),
    max_new_tokens: Number(lengthInput.value),
    top_k: 10,
    repair: document.querySelector("#repairInput").checked,
    constrain: document.querySelector("#constrainInput").checked
  };
}

function renderSwaras(tokens) {
  const swaras = tokens.filter((token) => token.startsWith("SWARA_")).slice(0, 48);
  swaraBoard.innerHTML = swaras.map((token, index) => {
    const tone = index % 8 === 0 ? "strong" : index % 4 === 0 ? "soft" : "base";
    return `<div class="swara-chip" data-tone="${tone}">${swaraNames[token] || token.replace("SWARA_", "")}</div>`;
  }).join("");
}

function renderResult(result) {
  const validation = result.validation;
  const audio = result.audio;
  scoreBadge.textContent = `Score ${validation.score}`;
  ragaBadge.textContent = validation.raga;
  validMetric.textContent = validation.valid ? "Yes" : "No";
  notesMetric.textContent = validation.notes;
  swarasMetric.textContent = validation.unique_swaras;
  secondsMetric.textContent = audio.seconds;
  lessonOutput.textContent = result.lesson;
  tokenOutput.textContent = result.tokens.join(" ");
  tokenCount.textContent = result.tokens.length;
  audioPlayer.src = result.files.wav;
  midiLink.href = result.files.midi;
  midiLink.removeAttribute("aria-disabled");
  renderSwaras(result.tokens);
}

async function generate(event) {
  event.preventDefault();
  generateButton.disabled = true;
  setStatus("Generating");
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadFromForm())
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error || "Generation failed");
    }
    renderResult(result);
    setStatus("Ready");
  } catch (error) {
    setStatus("Error");
    lessonOutput.textContent = error.message;
  } finally {
    generateButton.disabled = false;
  }
}

async function loadOptions() {
  try {
    const response = await fetch("/api/options");
    if (!response.ok) return;
    const options = await response.json();
    deviceBadge.textContent = options.device || "Local";
  } catch (error) {
    deviceBadge.textContent = "Local";
  }
}

temperatureInput.addEventListener("input", () => {
  temperatureValue.textContent = Number(temperatureInput.value).toFixed(2);
});

lengthInput.addEventListener("input", () => {
  lengthValue.textContent = lengthInput.value;
});

form.addEventListener("submit", generate);
loadOptions();
