/* static/js/main.js — Frontend logic for DocSummarizer */

// ── State ─────────────────────────────────────────────────────────────────
let selectedFile = null;

// ── File drag-and-drop ────────────────────────────────────────────────────
const dropZone  = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");

dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

dropZone.addEventListener("click", e => {
  if (e.target.closest("button")) return;
  fileInput.click();
});

function setFile(file) {
  selectedFile = file;
  document.querySelector(".drop-main").textContent = `📄 ${file.name}`;
  document.querySelector(".drop-sub").textContent =
    `${(file.size / 1024).toFixed(1)} KB · Click to change`;
}

// ── Submit ────────────────────────────────────────────────────────────────
async function submitSummary() {
  const rawText = document.getElementById("rawText").value.trim();
  const referenceSummary = document.getElementById("referenceSummary").value.trim();

  if (!selectedFile && !rawText) {
    showToast("Please upload a file or paste some text first.");
    return;
  }

  setLoading(true);

  const fd = new FormData();
  fd.append("length", "medium");  // Always use medium with adaptive compression
  if (referenceSummary) fd.append("reference_summary", referenceSummary);
  if (selectedFile) fd.append("file", selectedFile);
  else              fd.append("text", rawText);

  try {
    const resp = await fetch("/summarize", { method: "POST", body: fd });
    const data = await resp.json();
    
    console.log("Response data:", data);  // Debug log

    if (!resp.ok || data.error) {
      showToast(data.error || "Something went wrong. Please try again.");
      setLoading(false);
      return;
    }

    renderResult(data);
    
    // Clear inputs upon successful generation
    document.getElementById("upload").classList.add("hidden");
    document.getElementById("rawText").value = "";
    document.getElementById("referenceSummary").value = "";
    selectedFile = null;
    document.getElementById("fileInput").value = "";
    document.querySelector(".drop-main").textContent = "Drag & drop your file here";
    document.querySelector(".drop-sub").textContent = "PDF · DOCX · PPTX · PNG · JPG";

  } catch (err) {
    console.error(err);
    showToast("Error: " + err.message);
  } finally {
    setLoading(false);
  }
}

// ── Show upload form again ───────────────────────────────────────────────
function resetForm() {
  document.getElementById("upload").classList.remove("hidden");
  document.getElementById("resultCard").classList.add("hidden");
  document.getElementById("summaryOutput").textContent = "";
  document.getElementById("extractiveText").textContent = "";
  updateCharCount();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Text limit tracker ────────────────────────────────────────────────────
function updateCharCount() {
  const len = document.getElementById("rawText").value.length;
  document.getElementById("charCount").textContent = `${len} / 10000 characters`;
}

// ── Render result ─────────────────────────────────────────────────────────
function renderResult({ summary, top_sentences, stats, evaluation, factuality }) {
  console.log("Rendering with factuality:", factuality);
  console.log("Rendering with evaluation:", evaluation);
  // Always display as paragraph (natural abstractive output)
  document.getElementById("summaryOutput").textContent = summary;

  document.getElementById("statTokens").textContent =
    `Mode: ${stats.summary_mode} · ${stats.selected_sentence_count} extracted / ${stats.total_sentence_count} sentences`;

  // Populate Dashboard Stats
  document.getElementById("statOrigWords").textContent = stats.original_word_count;
  document.getElementById("statSummWords").textContent = stats.summary_word_count;
  document.getElementById("statCompression").textContent = stats.compression_ratio;
  document.getElementById("statTotalSents").textContent = stats.total_sentence_count;
  document.getElementById("statSelSents").textContent = stats.selected_sentence_count;
  
  if(document.getElementById("statChunkCount")) {
      document.getElementById("statChunkCount").textContent = stats.chunk_count;
      document.getElementById("statAvgTokens").textContent = stats.avg_chunk_tokens;
  }
  
  document.getElementById("statMode").textContent = stats.summary_mode;
  document.getElementById("statModel").textContent = stats.model_used;
  document.getElementById("statTime").textContent = stats.processing_time;

  document.getElementById("statPath").textContent = stats.summary_path;

  // Display factuality score with color coding
  if (factuality && factuality.score !== undefined) {
    const factScore = factuality.score;
    let factColor = '#00e5a0'; // Green
    if (factScore < 0.75) factColor = '#f59e0b'; // Orange
    if (factScore < 0.60) factColor = '#f43f5e'; // Red
    
    document.getElementById("statFactuality").innerHTML = 
      `<span style="color: ${factColor}; font-size: 1.1em;">${factScore.toFixed(3)}</span>`;
  } else {
    document.getElementById("statFactuality").textContent = '-';
  }

  // Display novel n-grams (abstractiveness metric)
  if (evaluation && evaluation.novel_bigrams !== undefined) {
    const novelty = evaluation.novel_bigrams;
    let noveltyColor = '#00e5a0';
    if (novelty < 40) noveltyColor = '#f59e0b';
    if (novelty < 20) noveltyColor = '#f43f5e';
    
    document.getElementById("statNovelty").innerHTML = 
      `<span style="color: ${noveltyColor}; font-size: 1.1em;">${novelty}%</span>`;
  } else {
    document.getElementById("statNovelty").textContent = '-';
  }

  // Display evaluation metrics with better formatting
  const evalMessage = document.getElementById("evalMessage");
  const evalStats = document.getElementById("evalStats");
  if (evaluation && Object.keys(evaluation).length > 0 && evaluation.rouge1 !== undefined) {
    document.getElementById("metricRouge1").innerHTML = `<span style="color: #00e5a0;">${evaluation.rouge1}</span>`;
    document.getElementById("metricRouge2").innerHTML = `<span style="color: #00e5a0;">${evaluation.rouge2}</span>`;
    document.getElementById("metricRougeL").innerHTML = `<span style="color: #00e5a0;">${evaluation.rougeL}</span>`;
    document.getElementById("metricBertF1").innerHTML = `<span style="color: #00e5a0;">${evaluation.bertscore_f1}</span>`;
    evalStats.style.display = "grid";
    evalMessage.style.display = "none";
  } else {
    evalStats.style.display = "none";
    evalMessage.style.display = "block";
  }

  const extractiveDiv = document.getElementById("extractiveText");
  if (typeof extractive_text !== "undefined" && extractive_text) {
    extractiveDiv.textContent = extractive_text;
  } else {
    extractiveDiv.textContent = top_sentences.join(" ");
  }

  const card = document.getElementById("resultCard");
  card.classList.remove("hidden");
  card.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Copy ──────────────────────────────────────────────────────────────────
function copySummary() {
  const text = document.getElementById("summaryOutput").textContent;
  navigator.clipboard.writeText(text).then(() => {
    showToast("Summary copied to clipboard!", "success");
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────
function setLoading(state) {
  const btn    = document.getElementById("summarizeBtn");
  const label  = document.getElementById("btnText");
  const loader = document.getElementById("btnLoader");
  btn.disabled = state;
  label.classList.toggle("hidden", state);
  loader.classList.toggle("hidden", !state);
}

function showToast(message, type = "error") {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.style.background = type === "success" ? "#00c47d" : "var(--accent3)";
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 3500);
}

// Ensure extractive_text is available in renderResult
window.renderResult = function (data) {
  // Support both old and new API responses
  const extractive_text = data.extractive_text || (data.top_sentences ? data.top_sentences.join(" ") : "");
  renderResult({ ...data, extractive_text });
};