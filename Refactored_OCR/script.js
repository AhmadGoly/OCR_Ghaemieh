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
  const fileSelectedBadge = document.getElementById("file-selected-badge");
  const fileNameDisplay = document.getElementById("file-name-display");
  const fileSizeDisplay = document.getElementById("file-size-display");
  const resultModelBadge = document.getElementById("result-model-badge");
  const resultOcrDuration = document.getElementById("result-ocr-duration");
  const resultLlmDuration = document.getElementById("result-llm-duration");
  const resultLlmContainer = document.getElementById("result-llm-container");
  const charWordCount = document.getElementById("char-word-count");
  const navUserActions = document.getElementById("nav-user-actions");

  // Token Elements
  const bannerTokenDisplay = document.getElementById("banner-token-display");
  const bannerCopyTokenBtn = document.getElementById("banner-copy-token-btn");
  const bannerCopyText = document.getElementById("banner-copy-text");
  const openTokenModalBtn = document.getElementById("open-token-modal-btn");
  const tokenModal = document.getElementById("token-modal");
  const closeTokenModalBtn = document.getElementById("close-token-modal-btn");
  const modalCancelBtn = document.getElementById("modal-cancel-btn");
  const modalTokenInput = document.getElementById("modal-token-input");
  const modalCopyTokenBtn = document.getElementById("modal-copy-token-btn");
  const copyBtnText = document.getElementById("copy-btn-text");
  const modalRegenerateTokenBtn = document.getElementById("modal-regenerate-token-btn");
  const curlCodeBlock = document.getElementById("curl-code-block");
  const copyCurlBtn = document.getElementById("copy-curl-btn");

  let selectedFile = null;
  let currentUserToken = null;
  let currentUser = null;

  // Initialize Lucide icons
  if (window.lucide) {
    lucide.createIcons();
  }

  // Check login state and retrieve user token
  checkUserSession();

  async function checkUserSession() {
    try {
      const res = await fetch("/api/auth/me");
      if (!res.ok) {
        // Not logged in -> redirect to login page immediately
        window.location.href = "/login";
        return;
      }

      currentUser = await res.json();

      // Retrieve user's API token
      await fetchUserToken();

      // Render top navigation bar
      let navHtml = `
        <div class="flex items-center space-x-2 space-x-reverse text-xs bg-zinc-900/80 px-3 py-1.5 rounded-xl border border-zinc-800">
          <i data-lucide="user-check" class="w-3.5 h-3.5 text-emerald-400"></i>
          <span class="text-slate-400">کاربر:</span>
          <span class="font-bold text-white">${currentUser.username}</span>
        </div>
        <button id="nav-token-btn" class="px-3.5 py-1.5 rounded-xl bg-red-950/60 hover:bg-red-900/60 border border-red-500/30 text-red-300 hover:text-white text-xs font-semibold transition flex items-center space-x-1.5 space-x-reverse shadow-md shadow-red-950/20">
          <i data-lucide="key" class="w-3.5 h-3.5"></i>
          <span>توکن اختصاصی</span>
        </button>
      `;

      if (currentUser.is_admin) {
        navHtml += `
          <a href="/admin" class="px-3.5 py-1.5 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-slate-300 hover:text-white text-xs font-semibold transition flex items-center space-x-1.5 space-x-reverse">
            <i data-lucide="shield" class="w-4 h-4 text-red-400"></i>
            <span>پنل مدیریت</span>
          </a>
        `;
      }

      navHtml += `
        <button id="nav-logout-btn" class="px-3.5 py-1.5 rounded-xl bg-zinc-800 hover:bg-zinc-700 text-slate-300 hover:text-white text-xs font-semibold transition flex items-center space-x-1.5 space-x-reverse">
          <i data-lucide="log-out" class="w-4 h-4"></i>
          <span>خروج</span>
        </button>
      `;

      navUserActions.innerHTML = navHtml;
      if (window.lucide) lucide.createIcons();

      // Logout handler
      document.getElementById("nav-logout-btn").addEventListener("click", async () => {
        await fetch("/api/auth/logout", { method: "POST" });
        window.location.href = "/login";
      });

      // Nav token button handler
      document.getElementById("nav-token-btn").addEventListener("click", () => {
        openModal();
      });

    } catch (e) {
      window.location.href = "/login";
    }
  }

  async function fetchUserToken() {
    try {
      const res = await fetch("/api/user/token");
      if (res.ok) {
        const data = await res.json();
        currentUserToken = data.token;
        updateTokenDisplay();
      }
    } catch (err) {
      console.error("Failed to load user token:", err);
    }
  }

  function updateTokenDisplay() {
    if (!currentUserToken) return;

    if (bannerTokenDisplay) {
      bannerTokenDisplay.textContent = currentUserToken;
    }
    if (modalTokenInput) {
      modalTokenInput.value = currentUserToken;
    }
    if (curlCodeBlock) {
      const origin = window.location.origin;
      curlCodeBlock.textContent = `curl -X POST "${origin}/ocr/image" \\
  -H "X-API-Key: ${currentUserToken}" \\
  -F "file=@document.jpg" \\
  -F "model=gemma4"`;
    }
  }

  // Modal Open & Close Handlers
  function openModal() {
    if (tokenModal) {
      tokenModal.classList.remove("hidden");
      if (window.lucide) lucide.createIcons();
    }
  }

  function closeModal() {
    if (tokenModal) {
      tokenModal.classList.add("hidden");
    }
  }

  if (openTokenModalBtn) {
    openTokenModalBtn.addEventListener("click", openModal);
  }
  if (closeTokenModalBtn) {
    closeTokenModalBtn.addEventListener("click", closeModal);
  }
  if (modalCancelBtn) {
    modalCancelBtn.addEventListener("click", closeModal);
  }

  // Copy Token Helper
  function copyTextToClipboard(text, btnElement, defaultLabel) {
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      if (btnElement) {
        btnElement.textContent = "کپی شد!";
        setTimeout(() => {
          btnElement.textContent = defaultLabel;
        }, 2000);
      }
    }).catch(() => {
      // Fallback
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      if (btnElement) {
        btnElement.textContent = "کپی شد!";
        setTimeout(() => {
          btnElement.textContent = defaultLabel;
        }, 2000);
      }
    });
  }

  if (bannerCopyTokenBtn) {
    bannerCopyTokenBtn.addEventListener("click", () => {
      copyTextToClipboard(currentUserToken, bannerCopyText, "کپی توکن");
    });
  }

  if (modalCopyTokenBtn) {
    modalCopyTokenBtn.addEventListener("click", () => {
      copyTextToClipboard(currentUserToken, copyBtnText, "کپی توکن");
    });
  }

  if (copyCurlBtn) {
    copyCurlBtn.addEventListener("click", () => {
      if (curlCodeBlock) {
        copyTextToClipboard(curlCodeBlock.textContent, copyCurlBtn, "کپی دستور");
      }
    });
  }

  // Regenerate Token Handler
  if (modalRegenerateTokenBtn) {
    modalRegenerateTokenBtn.addEventListener("click", async () => {
      const confirmed = confirm(
        "آیا از تولید مجدد توکن مطمئن هستید؟\n\nتوکن فعلی شما بلافاصله باطل شده و تمامی برنامه‌ها یا اسکریپت‌هایی که از آن استفاده می‌کنند باید به‌روز شوند."
      );
      if (!confirmed) return;

      modalRegenerateTokenBtn.disabled = true;
      modalRegenerateTokenBtn.innerHTML = `
        <i data-lucide="loader-2" class="w-3.5 h-3.5 animate-spin"></i>
        <span>در حال ایجاد...</span>
      `;
      if (window.lucide) lucide.createIcons();

      try {
        const res = await fetch("/api/user/token/regenerate", { method: "POST" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "خطا در تولید مجدد توکن");

        currentUserToken = data.token;
        updateTokenDisplay();
        alert("توکن تصادفی جدید با موفقیت صادر شد و برای حسابتان ثبت گردید.");
      } catch (err) {
        alert("خطا: " + err.message);
      } finally {
        modalRegenerateTokenBtn.disabled = false;
        modalRegenerateTokenBtn.innerHTML = `
          <i data-lucide="refresh-cw" class="w-3.5 h-3.5"></i>
          <span>تولید مجدد توکن تصادفی</span>
        `;
        if (window.lucide) lucide.createIcons();
      }
    });
  }

  // Toggle hints visibility
  toggleHintsButton.addEventListener("click", () => {
    const hints = document.querySelectorAll(".hint");
    const isHidden = hints.length > 0 && (hints[0].classList.contains("hidden") || hints[0].style.display === "none");
    hints.forEach((hint) => {
      if (isHidden) {
        hint.classList.remove("hidden");
        hint.style.display = "block";
      } else {
        hint.classList.add("hidden");
        hint.style.display = "none";
      }
    });
  });

  // Scale slider update
  scaleSlider.addEventListener("input", (e) => {
    scaleValue.textContent = parseFloat(e.target.value).toFixed(1);
  });

  // LLM toggle reveals secondary model
  llmToggle.addEventListener("change", (e) => {
    if (e.target.checked) {
      secondaryModelContainer.classList.remove("hidden");
    } else {
      secondaryModelContainer.classList.add("hidden");
    }
  });

  // File browser trigger
  fileBrowserButton.addEventListener("click", () => fileInput.click());
  uploadArea.addEventListener("click", (e) => {
    if (e.target !== fileBrowserButton && !fileBrowserButton.contains(e.target)) {
      fileInput.click();
    }
  });

  // Drag and Drop
  ["dragenter", "dragover"].forEach((eventName) => {
    uploadArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      uploadArea.classList.add("border-red-500", "bg-red-950/20");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    uploadArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      uploadArea.classList.remove("border-red-500", "bg-red-950/20");
    });
  });

  uploadArea.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    fileNameDisplay.textContent = file.name;
    fileSizeDisplay.textContent = formatBytes(file.size);
    fileSelectedBadge.classList.remove("hidden");
    submitButton.disabled = false;

    // Toggle PDF page inputs
    const isPdf = file.type === "application/pdf" || file.name.endsWith(".pdf");
    startPageInput.disabled = !isPdf;
    endPageInput.disabled = !isPdf;
  }

  function formatBytes(bytes, decimals = 2) {
    if (!+bytes) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
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

    if (selectedFile.type === "application/pdf" || selectedFile.name.endsWith(".pdf")) {
      if (startPageInput.value) formData.append("start_page", startPageInput.value);
      if (endPageInput.value) formData.append("end_page", endPageInput.value);
    }

    const endpoint = (selectedFile.type === "application/pdf" || selectedFile.name.endsWith(".pdf")) ? "/ocr/pdf" : "/ocr/image";

    // Show loading
    loadingIndicator.hidden = false;
    resultsArea.hidden = true;
    submitButton.disabled = true;
    imageResultsContainer.hidden = true;

    try {
      const headers = {};
      if (currentUserToken) {
        headers["X-API-Key"] = currentUserToken;
      }

      const response = await fetch(endpoint, {
        method: "POST",
        headers: headers,
        body: formData,
      });

      if (response.status === 401) {
        alert("نشست یا توکن کاربری شما منقضی شده است. لطفاً مجدداً وارد سیستم شوید.");
        window.location.href = "/login";
        return;
      }

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "خطا در پردازش فایل");
      }

      const result = await response.json();
      displayResults(result, !endpoint.includes("pdf"));
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
    let primaryModel = "";
    let ocrDur = 0;
    let llmDur = -1;

    if (Array.isArray(result)) {
      // PDF Results
      textOutput = result.map((page) => `--- صفحه ${page.page} ---\n${page.text}\n`).join("\n");
      if (result.length > 0) {
        primaryModel = result[0].ocr_model;
        ocrDur = result.reduce((acc, p) => acc + (p.ocr_duration || 0), 0).toFixed(2);
        llmDur = result.reduce((acc, p) => acc + (p.llm_duration > 0 ? p.llm_duration : 0), 0).toFixed(2);
      }
    } else {
      // Single Image Result
      textOutput = result.text || "";
      primaryModel = result.ocr_model;
      ocrDur = (result.ocr_duration || 0).toFixed(2);
      llmDur = result.llm_duration > 0 ? result.llm_duration.toFixed(2) : -1;

      if (isImage && result.original_image && result.processed_image) {
        imageResultsContainer.hidden = false;
        originalImage.src = `data:image/png;base64,${result.original_image}`;
        processedImage.src = `data:image/png;base64,${result.processed_image}`;
      }
    }

    ocrOutput.value = textOutput;

    // Update metrics
    resultModelBadge.textContent = primaryModel;
    resultOcrDuration.textContent = `${ocrDur} ثانیه`;

    if (llmDur > 0) {
      resultLlmContainer.hidden = false;
      resultLlmDuration.textContent = `${llmDur} ثانیه`;
    } else {
      resultLlmContainer.hidden = true;
    }

    // Word and character count
    const words = textOutput.trim() ? textOutput.trim().split(/\s+/).length : 0;
    const chars = textOutput.length;
    charWordCount.textContent = `${words} کلمه | ${chars} کاراکتر`;

    if (window.lucide) lucide.createIcons();
    window.scrollTo({ top: resultsArea.offsetTop - 50, behavior: "smooth" });
  }

  // Copy result text
  copyButton.addEventListener("click", () => {
    navigator.clipboard.writeText(ocrOutput.value).then(() => {
      const originalText = copyButton.querySelector("span").textContent;
      copyButton.querySelector("span").textContent = "کپی شد!";
      setTimeout(() => {
        copyButton.querySelector("span").textContent = originalText;
      }, 2000);
    });
  });

  // Download result text
  downloadButton.addEventListener("click", () => {
    const blob = new Blob([ocrOutput.value], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ocr_result_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  });
});
