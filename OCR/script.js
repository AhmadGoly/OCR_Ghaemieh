document.addEventListener("DOMContentLoaded", () => {
  // Element selections
  const uploadArea = document.getElementById("upload-area");
  const fileBrowserButton = document.getElementById("file-browser-button");
  const fileInput = document.getElementById("file-input");
  const submitButton = document.getElementById("submit-button");
  const loadingIndicator = document.getElementById("loading-indicator");
  const resultsArea = document.getElementById("results-area");
  const ocrOutput = document.getElementById("ocr-output");
  const imageResultsContainer = document.getElementById(
    "image-results-container"
  );
  const originalImage = document.getElementById("original-image");
  const processedImage = document.getElementById("processed-image");
  const copyButton = document.getElementById("copy-button");
  const downloadButton = document.getElementById("download-button");
  const scaleSlider = document.getElementById("scale-slider");
  const scaleValue = document.getElementById("scale-value");
  const toggleHints = document.getElementById("toggle-hints");
  toggleHints.addEventListener("click", () => {
    document.body.classList.toggle("show-hints");
    toggleHints.textContent = document.body.classList.contains("show-hints")
      ? "مخفی کردن راهنما"
      : "نمایش راهنما";
  });
  let selectedFile = null;

  // --- File Upload Handling ---
  fileBrowserButton.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    handleFileSelect(e.target.files[0]);
  });

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
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileSelect(file);
    }
  });

  function handleFileSelect(file) {
    if (!file) return;

    const acceptedTypes = [
      "image/png",
      "image/jpeg",
      "image/tiff",
      "application/pdf",
    ];
    if (!acceptedTypes.includes(file.type)) {
      alert(
        "فرمت فایل پشتیبانی نمی‌شود. لطفاً یک فایل تصویر (PNG, JPEG, TIFF) یا PDF انتخاب کنید."
      );
      return;
    }

    selectedFile = file;
    uploadArea.querySelector("p").textContent = `فایل انتخاب شده: ${file.name}`;
    submitButton.disabled = false;
  }

  // --- Options Handling ---
  scaleSlider.addEventListener("input", (e) => {
    scaleValue.textContent = e.target.value;
  });

  // --- Form Submission ---
  submitButton.addEventListener("click", async () => {
    if (!selectedFile) {
      alert("لطفاً یک فایل را انتخاب کنید.");
      return;
    }

    const isPdf = selectedFile.type === "application/pdf";
    const endpoint = isPdf ? "/ocr/pdf" : "/ocr/image";

    const formData = new FormData();
    formData.append("file", selectedFile);

    // Append options from the UI
    formData.append("model", document.getElementById("model-select").value);
    formData.append("lang", document.getElementById("lang-select").value);
    formData.append(
      "preprocess",
      document.getElementById("preprocess-toggle").checked
    );
    formData.append(
      "contrast",
      document.getElementById("contrast-toggle").checked
    );
    formData.append(
      "crop_whitespaces",
      document.getElementById("crop-toggle").checked
    );
    formData.append("scale", scaleSlider.value);
    formData.append("use_llm", document.getElementById("llm-toggle").checked);

    // PDF-specific options
    if (isPdf) {
      const startPage = document.getElementById("start-page").value;
      const endPage = document.getElementById("end-page").value;
      if (startPage) formData.append("start_page", startPage);
      if (endPage) formData.append("end_page", endPage);
    }

    // --- API Call and Result Handling ---
    try {
      loadingIndicator.hidden = false;
      resultsArea.hidden = true;
      submitButton.disabled = true;

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(
          errorData.detail || `HTTP error! status: ${response.status}`
        );
      }

      const data = await response.json();

      let fullText = "";
      if (isPdf) {
        imageResultsContainer.hidden = true;
        // For PDF, data is an array of page results
        data.sort((a, b) => a.page - b.page); // Ensure pages are in order
        fullText = data
          .map((page) => `--- صفحه ${page.page} ---\n${page.text}`)
          .join("\n\n");
      } else {
        // For Image, data is a single result object
        fullText = data.text;
        originalImage.src = `data:image/png;base64,${data.original_image}`;
        processedImage.src = `data:image/png;base64,${data.processed_image}`;
        imageResultsContainer.hidden = false;
      }

      ocrOutput.textContent = fullText;
      resultsArea.hidden = false;
    } catch (error) {
      console.error("Error during OCR processing:", error);
      alert(`خطا در پردازش: ${error.message}`);
    } finally {
      loadingIndicator.hidden = true;
      submitButton.disabled = false;
    }
  });

  // --- Results Actions ---
  copyButton.addEventListener("click", () => {
    navigator.clipboard
      .writeText(ocrOutput.textContent)
      .then(() => alert("متن در کلیپ‌بورد کپی شد!"))
      .catch((err) => console.error("Could not copy text: ", err));
  });

  downloadButton.addEventListener("click", () => {
    const text = ocrOutput.textContent;
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ocr_result_${new Date().toISOString()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
});
