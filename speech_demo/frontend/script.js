/* Drag-and-drop upload behavior */
const dropZone = document.getElementById("dropZone");
const audioFile = document.getElementById("audioFile");
const fileLabel = document.getElementById("fileLabel");

function setSelectedFile(file) {
  if (file) {
    fileLabel.textContent = file.name;
    dropZone.classList.add("has-file");
  } else {
    fileLabel.textContent = "Drop audio file or click to browse";
    dropZone.classList.remove("has-file");
  }
}

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");

  const files = e.dataTransfer.files;
  if (files.length && files[0].type.startsWith("audio/")) {
    audioFile.files = files;
    setSelectedFile(files[0]);
  }
});

audioFile.addEventListener("change", () => {
  if (audioFile.files.length) {
    setSelectedFile(audioFile.files[0]);
  } else {
    setSelectedFile(null);
  }
});

/*
  Word-level diff highlight
  - Compares hypothesis words against reference words.
  - Returns HTML string with <span class="word ok|err"> tokens.
  - Uses LCS alignment compatible with standard WER edits.
*/
function diffHighlight(hypothesis, reference) {
  const hypWords = hypothesis.trim().toLowerCase().replace(/[.,!?;:"']/g, "").split(/\s+/);
  const refWords = reference.trim().toLowerCase().replace(/[.,!?;:"']/g, "").split(/\s+/);
  const origHyp = hypothesis.trim().split(/\s+/);

  const H = hypWords.length;
  const R = refWords.length;
  const dp = Array.from({ length: H + 1 }, () => new Int32Array(R + 1));

  for (let i = 1; i <= H; i++) {
    for (let j = 1; j <= R; j++) {
      dp[i][j] = hypWords[i - 1] === refWords[j - 1]
        ? dp[i - 1][j - 1] + 1
        : Math.max(dp[i - 1][j], dp[i][j - 1]);
    }
  }

  const matched = new Array(H).fill(false);
  let i = H;
  let j = R;

  while (i > 0 && j > 0) {
    if (hypWords[i - 1] === refWords[j - 1]) {
      matched[i - 1] = true;
      i--;
      j--;
    } else if (dp[i - 1][j] > dp[i][j - 1]) {
      i--;
    } else {
      j--;
    }
  }

  return origHyp
    .map((word, idx) => `<span class="word ${matched[idx] ? "ok" : "err"}">${word}</span>`)
    .join(" ");
}

/* Main evaluation request */
async function evaluateAudio() {
  const file = document.getElementById("audioFile").files[0];
  const domain = document.getElementById("domain").value;
  const reference = document.getElementById("reference").value.trim();

  if (!file) {
    alert("Please select an audio file.");
    return;
  }

  if (!reference) {
    alert("Please paste the reference transcript.");
    return;
  }

  const btn = document.getElementById("evalBtn");
  btn.disabled = true;
  btn.classList.add("loading");

  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("domain", domain);
    formData.append("reference", reference);

    const response = await fetch("http://127.0.0.1:8000/evaluate", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    document.getElementById("baseWer").textContent = `${data.base_wer}%`;
    document.getElementById("loraWer").textContent = `${data.lora_wer}%`;
    document.getElementById("improvement").textContent = `${data.improvement}%`;

    const baseEl = document.getElementById("baseText");
    const loraEl = document.getElementById("loraText");

    if (reference) {
      baseEl.innerHTML = diffHighlight(data.base_transcript, reference);
      loraEl.innerHTML = diffHighlight(data.lora_transcript, reference);
    } else {
      baseEl.textContent = data.base_transcript;
      loraEl.textContent = data.lora_transcript;
    }

    const panel = document.getElementById("resultsPanel");
    panel.style.animation = "none";
    panel.offsetHeight;
    panel.style.animation = "";
  } catch (err) {
    alert("Evaluation failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.classList.remove("loading");
  }
}
