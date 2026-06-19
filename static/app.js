const form = document.getElementById("searchForm");
const statusText = document.getElementById("statusText");
const loadingState = document.getElementById("loadingState");
const loadingText = document.getElementById("loadingText");
const errorBanner = document.getElementById("errorBanner");
const emptyState = document.getElementById("emptyState");
const resultsSection = document.getElementById("resultsSection");
const runButton = document.getElementById("runButton");
const resultStatusChip = document.getElementById("resultStatusChip");
const resultImage = document.getElementById("resultImage");
const detailsToggle = document.getElementById("detailsToggle");
const detailsPanel = document.getElementById("detailsPanel");
const summaryText = document.getElementById("summaryText");
const notesList = document.getElementById("notesList");
const metricsList = document.getElementById("metricsList");
const sourceLine = document.getElementById("sourceLine");
const confidenceValue = document.getElementById("confidenceValue");
const periodValue = document.getElementById("periodValue");
const durationValue = document.getElementById("durationValue");
const radiusValue = document.getElementById("radiusValue");
const resultReason = document.getElementById("resultReason");
const targetField = document.querySelector('[data-mode-field="target"]');
const fileField = document.querySelector('[data-mode-field="file"]');
const sourceRadios = document.querySelectorAll('input[name="source_mode"]');
const targetInput = document.getElementById("targetId");
const fileInput = document.getElementById("lightcurveFile");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setHidden(element, hidden) {
  element.classList.toggle("hidden", hidden);
}

function formatValue(value, suffix = "") {
  if (value === null || value === undefined || value === "") {
    return "--";
  }
  return `${value}${suffix}`;
}

function syncSourceMode() {
  const selected = document.querySelector('input[name="source_mode"]:checked')?.value || "target";
  const isFileMode = selected === "file";
  setHidden(targetField, isFileMode);
  setHidden(fileField, !isFileMode);
  targetInput.disabled = isFileMode;
  fileInput.disabled = !isFileMode;
}

function setLoading(isLoading, message = "Running BLS search and building the dashboard...") {
  setHidden(loadingState, !isLoading);
  loadingText.textContent = message;
  runButton.disabled = isLoading;
  statusText.textContent = isLoading ? "Running search..." : "Ready";
}

function clearError() {
  setHidden(errorBanner, true);
  errorBanner.textContent = "";
}

function showError(message) {
  errorBanner.textContent = message;
  setHidden(errorBanner, false);
  resultsSection.classList.add("hidden");
  emptyState.classList.remove("hidden");
}

function buildMetricRow(label, value) {
  const row = document.createElement("div");
  row.className = "metric-row";
  row.innerHTML = `<span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>`;
  return row;
}

function renderDetails(data) {
  summaryText.textContent = (data.summary_lines || []).join("\n");

  notesList.innerHTML = "";
  if ((data.notes || []).length) {
    data.notes.forEach((note) => {
      const li = document.createElement("li");
      li.textContent = note;
      notesList.appendChild(li);
    });
  } else {
    const li = document.createElement("li");
    li.textContent = "No additional notes were returned.";
    notesList.appendChild(li);
  }

  metricsList.innerHTML = "";
  const metrics = data.metrics || {};
  if (Object.keys(metrics).length === 0) {
    metricsList.appendChild(buildMetricRow("Metrics", "No extra metrics available."));
  } else {
    for (const [key, value] of Object.entries(metrics)) {
      metricsList.appendChild(buildMetricRow(key, Array.isArray(value) ? JSON.stringify(value) : String(value)));
    }
  }
}

function renderResult(data) {
  emptyState.classList.add("hidden");
  resultsSection.classList.remove("hidden");
  sourceLine.textContent = `Source: ${data.display_source || "Unknown"}`;
  resultStatusChip.textContent = data.status || "Result";
  resultImage.src = data.figure_data_uri;
  resultImage.alt = `Transit dashboard for ${data.display_source || "the current analysis"}`;

  confidenceValue.textContent = formatValue(data.confidence_percent, "%");
  const candidate = data.candidate || {};
  periodValue.textContent = formatValue(candidate.period_days, " d");
  durationValue.textContent = formatValue(candidate.duration_hours, " hr");
  radiusValue.textContent =
    candidate.planet_radius_rearth !== null && candidate.planet_radius_rearth !== undefined
      ? `${candidate.planet_radius_rearth} R⊕`
      : "--";
  resultReason.textContent = data.reason || "No additional explanation was returned.";

  renderDetails(data);
  setHidden(detailsToggle, false);
  setHidden(detailsPanel, true);
  detailsToggle.textContent = "Show More Details";
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

sourceRadios.forEach((radio) => {
  radio.addEventListener("change", syncSourceMode);
});

detailsToggle.addEventListener("click", () => {
  const isHidden = detailsPanel.classList.contains("hidden");
  setHidden(detailsPanel, !isHidden);
  detailsToggle.textContent = isHidden ? "Hide More Details" : "Show More Details";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();
  setLoading(true);

  const formData = new FormData(form);

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    if (!response.ok || !data.success) {
      throw new Error(data.error || "The analysis request failed.");
    }

    renderResult(data);
    statusText.textContent = "Search complete";
  } catch (error) {
    showError(error.message || "An unexpected error occurred.");
    statusText.textContent = "Search failed";
  } finally {
    setLoading(false);
  }
});

syncSourceMode();
