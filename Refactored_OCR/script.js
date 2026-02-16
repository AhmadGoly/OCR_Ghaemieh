document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("file-input");
  const fileBrowserButton = document.getElementById("file-browser-button");
  const uploadArea = document.getElementById("upload-area");
  const submitButton = document.getElementById("submit-button");
  const modelSelect = document.getElementById("model-select");
  const langSelect = document.getElementById("lang-select");
  const preprocessToggle = document.getElementById("preprocess-toggle");
  const contrastToggle = document.getElementById("contrast-toggle");
  const cropToggle = document.getElementById("crop-toggle");
  const llmToggle = document.getElementById("llm-toggle");
  const secondaryModelContainer = document.getElementById("secondary-model-container");
  const secondaryModelSelect = document.getElementById("secondary-model-select");
  const scaleSlider = document.getElementById("scale-slider");
  const scaleValue = document.getElementById("scale-value");
  const startPageInput = document.getElementById("start-page");
  const endPageInput = document.getElementById("end-page");
  const resultsArea = document.getElementById("results-area");
  const ocrOutput = document.getElementById("ocr-output");
  const loadingIndicator = document.getElementById("loading-indicator");
  const copyButton = document.getElementById("copy-button");
  const downloadButton = document.getElementById("download-button");
  const toggleHintsButton = document.getElementById("toggle-hints");
  const imageResultsContainer = document.getElementById("image-results-container");
  const originalImage = document.getElementById("original-image");
  const processedImage = document.getElementById("processed-image");

  let selectedFile = null;

  // Toggle hints visibility
  toggleHintsButton.addEventListener("click", () => {
    const hints = document.querySelectorAll(".hint");
    const isHidden = hints[0].style.display === "none" || hints[0].style.display === "";
    hints.forEach((hint) => {
      hint.style.display = isHidden ? "block" : "none";
    });
    toggleHintsButton.textContent = isHidden ? "مخفی‌سازی راهنما" : "نمایش راهنما";
  });

  // Scale slider update
  scaleSlider.addEventListener("input", () => {
    scaleValue.textContent = scaleSlider.value;
  });

  // LLM Toggle listener to show/hide secondary model
  llmToggle.addEventListener("change", () => {
    secondaryModelContainer.style.display = llmToggle.checked ? "block" : "none";
  });

  // File selection
  fileBrowserButton.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  // Drag and drop
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });

  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
  });

  uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  function handleFile(file) {
    selectedFile = file;
    uploadArea.querySelector("p").textContent = `فایل انتخاب شده: ${file.name}`;
    submitButton.disabled = false;
  }

  // Submit request
  submitButton.addEventListener("click", async () => {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("model", modelSelect.value);
    formData.append("lang", langSelect.value);
    formData.append("preprocess", preprocessToggle.checked);
    formData.append("contrast", contrastToggle.checked);
    formData.append("crop_whitespaces", cropToggle.checked);
    formData.append("scale", scaleSlider.value);
    formData.append("use_llm", llmToggle.checked);

    if (llmToggle.checked && secondaryModelSelect.value) {
        formData.append("secondary_model", secondaryModelSelect.value);
    }

    if (selectedFile.type === "application/pdf") {
      if (startPageInput.value) formData.append("start_page", startPageInput.value);
      if (endPageInput.value) formData.append("end_page", endPageInput.value);
    }

    const endpoint = selectedFile.type === "application/pdf" ? "/ocr/pdf" : "/ocr/image";

    // Show loading
    loadingIndicator.hidden = false;
    resultsArea.hidden = true;
    submitButton.disabled = true;
    imageResultsContainer.hidden = true;

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "خطا در پردازش فایل");
      }

      const result = await response.json();
      displayResults(result, selectedFile.type !== "application/pdf");
    } catch (error) {
      alert("خطا: " + error.message);
    } finally {
      loadingIndicator.hidden = true;
      submitButton.disabled = false;
    }
  });

  function displayResults(result, isImage) {
    resultsArea.hidden = false;
    let textOutput = "";

    if (Array.isArray(result)) {
      // PDF Results
      textOutput = result.map((page) => `--- صفحه ${page.page} ---\n${page.text}\n`).join("\n");
    } else {
      // Image Result
      textOutput = result.text;
      if (isImage && result.original_image && result.processed_image) {
        imageResultsContainer.hidden = false;
        originalImage.src = `data:image/png;base64,${result.original_image}`;
        processedImage.src = `data:image/png;base64,${result.processed_image}`;
      }
    }

    ocrOutput.textContent = textOutput;
    window.scrollTo({ top: resultsArea.offsetTop, behavior: "smooth" });
  }

  // Copy results
  copyButton.addEventListener("click", () => {
    navigator.clipboard.writeText(ocrOutput.textContent);
    alert("متن کپی شد!");
  });

  // Download results
  downloadButton.addEventListener("click", () => {
    const blob = new Blob([ocrOutput.textContent], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ocr_result.txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
});
