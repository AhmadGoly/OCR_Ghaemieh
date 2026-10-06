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

  // Error Modal Elements
  const errorModal = document.getElementById("error-modal");
  const errorModalMessage = document.getElementById("error-modal-message");
  const errorModalCode = document.getElementById("error-modal-code");
  const errorModalDetails = document.getElementById("error-modal-details");
  const closeErrorModalBtn = document.getElementById("close-error-modal-btn");
  const errorModalCloseBtn = document.getElementById("error-modal-close-btn");
  const copyErrorCodeBtn = document.getElementById("copy-error-code-btn");
  const copyErrorBtnText = document.getElementById("copy-error-btn-text");
  const toggleErrorDetailsBtn = document.getElementById("toggle-error-details-btn");
  const errorDetailsChevron = document.getElementById("error-details-chevron");
  const appVersionBadge = document.getElementById("app-version-badge");

  // Background Book Tasks Elements
  const submitBookTaskButton = document.getElementById("submit-book-task-button");
  const cooldownInput = document.getElementById("cooldown-input");
  const bookTasksSection = document.getElementById("book-tasks-section");
  const bookTasksList = document.getElementById("book-tasks-list");
  const tasksLimitSelect = document.getElementById("tasks-limit-select");
  const refreshTasksBtn = document.getElementById("refresh-tasks-btn");
  const activeBookTasksBadge = document.getElementById("active-book-tasks-badge");

  // Task Detail Modal Elements
  const taskDetailModal = document.getElementById("task-detail-modal");
  const taskDetailTitle = document.getElementById("task-detail-title");
  const taskDetailSubtitle = document.getElementById("task-detail-subtitle");
  const taskDetailPagesBody = document.getElementById("task-detail-pages-body");
  const taskDetailFooterSummary = document.getElementById("task-detail-footer-summary");
  const closeTaskDetailBtn = document.getElementById("close-task-detail-btn");
  const taskDetailCloseFooterBtn = document.getElementById("task-detail-close-footer-btn");

  let selectedFile = null;
  let currentUserToken = null;
  let currentUser = null;
  let tasksPollTimer = null;
  let openDetailTaskId = null;

  // Initialize Lucide icons
  if (window.lucide) {
    lucide.createIcons();
  }

  // Check login state and retrieve user token
  checkUserSession();

  // Synchronize model availability and disable inactive models
  syncModelAvailability();

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

      // Load user's background book tasks
      await fetchBookTasks();

      // Render top navigation bar
      let navHtml = `
        <div class="flex items-center space-x-2 space-x-reverse text-xs bg-zinc-900/80 px-3 py-1.5 rounded-xl border border-zinc-800">
          <i data-lucide="user-check" class="w-3.5 h-3.5 text-emerald-400"></i>
          <span class="text-slate-400">کاربر:</span>
          <span class="font-bold text-white">${currentUser.username}</span>
          <span class="text-[10px] font-semibold px-2 py-0.5 rounded-full ${currentUser.is_admin ? 'bg-red-500/20 text-red-300 border border-red-500/30' : 'bg-zinc-800 text-slate-300 border border-zinc-700'}">${currentUser.is_admin ? 'مدیر' : 'عادی'}</span>
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
      curlCodeBlock.textContent = `# ۱. استخراج فوری تصویر:\ncurl -X POST "${origin}/ocr/image" \\\n  -H "X-API-Key: ${currentUserToken}" \\\n  -F "file=@document.jpg" \\\n  -F "model=gemma4"\n\n# ۲. ثبت کتاب در صف پس‌زمینه:\ncurl -X POST "${origin}/api/tasks/book" \\\n  -H "X-API-Key: ${currentUserToken}" \\\n  -F "file=@book.pdf" \\\n  -F "model=gemma4" \\\n  -F "cooldown_seconds=1.0"`;
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
        showErrorModal({
          status: 400,
          rawDetail: err.message,
          customMessage: "متأسفانه ایجاد توکن تصادفی جدید با خطا مواجه شد. لطفاً به مدیر سایت اطلاع دهید."
        });
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

  // Global Clipboard Paste Listener (Ctrl+V) for Screenshots
  window.addEventListener("paste", (e) => {
    const items = (e.clipboardData || window.clipboardData)?.items;
    if (!items) return;

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.type && item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (file) {
          e.preventDefault();
          const ext = item.type.split("/")[1] || "png";
          const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
          const pastedFile = new File([file], `screenshot_${timestamp}.${ext}`, { type: item.type });

          handleFileSelected(pastedFile);

          if (fileNameDisplay) {
            fileNameDisplay.textContent = `اسکرین‌شات الصاق‌شده (${pastedFile.name})`;
          }

          if (uploadArea) {
            uploadArea.classList.add("border-emerald-500", "bg-emerald-950/20");
            setTimeout(() => {
              uploadArea.classList.remove("border-emerald-500", "bg-emerald-950/20");
            }, 1500);
          }
          break;
        }
      }
    }
  });

  function handleFileSelected(file) {
    selectedFile = file;
    fileNameDisplay.textContent = file.name;
    fileSizeDisplay.textContent = formatBytes(file.size);
    fileSelectedBadge.classList.remove("hidden");
    submitButton.disabled = false;
    if (submitBookTaskButton) {
      submitBookTaskButton.disabled = false;
    }

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
        showErrorModal({
          status: 401,
          rawDetail: "Unauthorized: Session or token expired.",
          errorCode: "ERR-401"
        });
        setTimeout(() => {
          window.location.href = "/login";
        }, 3000);
        return;
      }

      if (!response.ok) {
        let errorDetail = "خطا در پردازش فایل";
        try {
          const errorData = await response.json();
          errorDetail = errorData.detail || errorDetail;
        } catch (_) {
          try {
            errorDetail = await response.text();
          } catch (__) {}
        }
        showErrorModal({
          status: response.status,
          rawDetail: errorDetail
        });
        return;
      }

      const result = await response.json();
      displayResults(result, !endpoint.includes("pdf"));
    } catch (error) {
      showErrorModal({
        status: 0,
        rawDetail: error.message || String(error)
      });
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

  // Multi-Format Export Handler for Active Result (TXT, MD, HTML, JSON)
  const exportDropdownBtn = document.getElementById("export-dropdown-btn");
  const exportDropdownMenu = document.getElementById("export-dropdown-menu");
  const exportDropdownChevron = document.getElementById("export-dropdown-chevron");
  const exportFormatButtons = document.querySelectorAll(".export-format-opt");

  function closeExportDropdown() {
    if (exportDropdownMenu) {
      exportDropdownMenu.classList.add("hidden");
    }
    if (exportDropdownChevron) {
      exportDropdownChevron.classList.remove("rotate-180");
    }
  }

  function toggleExportDropdown() {
    if (!exportDropdownMenu) return;
    const isHidden = exportDropdownMenu.classList.contains("hidden");
    if (isHidden) {
      exportDropdownMenu.classList.remove("hidden");
      if (exportDropdownChevron) exportDropdownChevron.classList.add("rotate-180");
    } else {
      closeExportDropdown();
    }
  }

  if (exportDropdownBtn) {
    exportDropdownBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleExportDropdown();
    });
  }

  document.addEventListener("click", (e) => {
    if (exportDropdownMenu && !exportDropdownMenu.contains(e.target) && e.target !== exportDropdownBtn) {
      closeExportDropdown();
    }
  });

  function triggerDownloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function exportActiveResult(format) {
    const text = ocrOutput ? ocrOutput.value : "";
    if (!text || !text.trim()) {
      alert("متنی برای دانلود وجود ندارد. لطفاً ابتدا سند یا تصویری را جهت استخراج ارسال فرمایید.");
      return;
    }

    const rawBase = selectedFile ? selectedFile.name.replace(/\.[^/.]+$/, "") : "ocr_result";
    const safeBase = rawBase.replace(/[^a-zA-Z0-9_\u0600-\u06FF\s-]/g, "_").trim() || "ocr_result";
    const timestamp = Date.now();
    const modelName = (resultModelBadge ? resultModelBadge.textContent : "") || "OCR Engine";
    const ocrTime = (resultOcrDuration ? resultOcrDuration.textContent : "") || "--";
    const llmTime = (resultLlmDuration && !resultLlmContainer?.hidden ? resultLlmDuration.textContent : "") || "";

    if (format === "txt") {
      const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
      triggerDownloadBlob(blob, `${safeBase}_${timestamp}.txt`);
    } else if (format === "md") {
      const mdContent = [
        `# خروجی استخراج متن قائمیه: ${safeBase}`,
        "",
        `- **نام سند**: \`${selectedFile ? selectedFile.name : safeBase}\``,
        `- **موتور استخراج متن**: \`${modelName}\`` + (llmTime ? ` + LLM (${llmTime})` : ""),
        `- **مدت زمان OCR**: \`${ocrTime}\``,
        `- **تاریخ استخراج**: \`${new Date().toLocaleString("fa-IR")}\``,
        `- **تعداد کاراکترها**: \`${text.length}\``,
        "",
        "---",
        "",
        text,
        ""
      ].join("\n");
      const blob = new Blob([mdContent], { type: "text/markdown;charset=utf-8" });
      triggerDownloadBlob(blob, `${safeBase}_${timestamp}.md`);
    } else if (format === "html") {
      const escapedText = escapeHtml(text).replace(/\n/g, "<br>\n");
      const htmlDoc = `<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${escapeHtml(safeBase)} - خروجی OCR قائمیه</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Vazirmatn', 'Tahoma', sans-serif;
            background: #f8fafc;
            color: #0f172a;
            margin: 0;
            padding: 32px 20px;
            line-height: 2.2;
            direction: rtl;
        }
        .container {
            max-width: 840px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
            padding: 36px 44px;
            border: 1px solid #e2e8f0;
        }
        .header {
            border-bottom: 2px solid #e11d48;
            padding-bottom: 16px;
            margin-bottom: 24px;
        }
        .header h1 {
            font-size: 20px;
            color: #be123c;
            margin: 0 0 10px 0;
        }
        .meta {
            font-size: 12px;
            color: #64748b;
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
        }
        .meta strong { color: #334155; }
        .content {
            font-size: 15px;
            color: #1e293b;
            text-align: justify;
            white-space: normal;
            line-height: 2.3;
        }
        @media print {
            body { background: #fff; padding: 0; }
            .container { box-shadow: none; border: none; padding: 0; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>${escapeHtml(selectedFile ? selectedFile.name : safeBase)}</h1>
            <div class="meta">
                <span>موتور OCR: <strong>${escapeHtml(modelName)}</strong></span>
                <span>زمان پردازش: <strong>${escapeHtml(ocrTime)}</strong></span>
                <span>تعداد کاراکتر: <strong>${text.length}</strong></span>
                <span>تاریخ: <strong>${new Date().toLocaleString("fa-IR")}</strong></span>
            </div>
        </div>
        <div class="content">
            ${escapedText}
        </div>
    </div>
</body>
</html>`;
      const blob = new Blob([htmlDoc], { type: "text/html;charset=utf-8" });
      triggerDownloadBlob(blob, `${safeBase}_${timestamp}.html`);
    } else if (format === "json") {
      const payload = {
        document_name: selectedFile ? selectedFile.name : safeBase,
        extracted_at: new Date().toISOString(),
        primary_model: modelName,
        ocr_duration: ocrTime,
        llm_duration: llmTime,
        character_count: text.length,
        word_count: text.trim().split(/\s+/).length,
        text: text
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
      triggerDownloadBlob(blob, `${safeBase}_${timestamp}.json`);
    }
  }

  exportFormatButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const fmt = btn.getAttribute("data-export-format") || "txt";
      exportActiveResult(fmt);
      closeExportDropdown();
    });
  });

  if (downloadButton) {
    downloadButton.addEventListener("click", () => {
      exportActiveResult("txt");
    });
  }

  // Synchronize model availability and disable inactive models in select menus
  async function syncModelAvailability() {
    try {
      const res = await fetch("/health/config");
      if (!res.ok) return;
      const configData = await res.json();

      if (configData.version && appVersionBadge) {
        appVersionBadge.textContent = "نسخه " + configData.version;
      }

      const modelsEnabled = configData.models_enabled || {};
      applyModelAvailability(modelSelect, modelsEnabled, false);
      applyModelAvailability(secondaryModelSelect, modelsEnabled, true);
    } catch (err) {
      console.warn("Failed to synchronize model availability:", err);
    }
  }

  function applyModelAvailability(selectElement, modelsEnabled, allowEmpty) {
    if (!selectElement) return;
    const options = Array.from(selectElement.options);
    let selectedOptionValid = false;

    options.forEach((opt) => {
      const val = opt.value;
      if (!val) {
        if (opt.selected) selectedOptionValid = true;
        return;
      }

      const isEnabled = modelsEnabled[val] !== false;
      opt.disabled = !isEnabled;

      let baseText = opt.getAttribute("data-original-title");
      if (!baseText) {
        baseText = opt.textContent.replace(/\s*\((غیرفعال|غیرفعال در سرور)\)/g, "").trim();
        opt.setAttribute("data-original-title", baseText);
      }

      if (!isEnabled) {
        opt.textContent = `${baseText} (غیرفعال)`;
        opt.classList.add("text-zinc-500", "bg-zinc-950");
      } else {
        opt.textContent = baseText;
        opt.classList.remove("text-zinc-500", "bg-zinc-950");
        if (opt.selected) {
          selectedOptionValid = true;
        }
      }
    });

    // If currently selected option is disabled, automatically fallback to first enabled option
    if (!selectedOptionValid) {
      const firstEnabled = options.find((opt) => !opt.disabled);
      if (firstEnabled) {
        selectElement.value = firstEnabled.value;
      }
    }
  }

  // User-Friendly Persian Error Modal Handler
  function showErrorModal({ status = 0, rawDetail = "", customMessage = "", errorCode = "" }) {
    if (!errorModal) {
      alert(customMessage || rawDetail || "خطایی رخ داده است.");
      return;
    }

    let friendlyMessage = customMessage;
    let code = errorCode;
    const detailStr = typeof rawDetail === "string" ? rawDetail : JSON.stringify(rawDetail || "");
    const lowerDetail = detailStr.toLowerCase();

    if (!friendlyMessage) {
      if (status === 401) {
        friendlyMessage = "اعتبار نشست کاربری یا کلید API شما منقضی شده است. لطفاً مجدداً وارد سامانه شوید.";
        code = code || "ERR-401";
      } else if (status === 403) {
        friendlyMessage = "حساب کاربری شما اجازه دسترسی به این بخش را ندارد.";
        code = code || "ERR-403";
      } else if (status === 413) {
        friendlyMessage = "حجم فایل ارسالی بیشتر از حد مجاز سرور است. لطفاً فایلی با حجم کمتر بارگذاری فرمایید.";
        code = code || "ERR-413";
      } else if (lowerDetail.includes("is not active on this server") || (status === 400 && lowerDetail.includes("model"))) {
        friendlyMessage = "مدل استخراج متن انتخابی در حال حاضر روی سرور فعال نیست. لطفاً مدل دیگری را انتخاب کرده یا با مدیر سایت تماس حاصل فرمایید.";
        code = code || "ERR-400-MDL";
      } else if (lowerDetail.includes("cuda") || lowerDetail.includes("out of memory") || lowerDetail.includes("memory")) {
        friendlyMessage = "منابع پردازشی سرور موقتاً تکمیل است. لطفاً چند لحظه بعد مجدداً تلاش کرده یا از مدل‌های دیگر استفاده کنید.";
        code = code || "ERR-503-MEM";
      } else if (lowerDetail.includes("connection refused") || lowerDetail.includes("connect") || status === 502 || status === 503) {
        friendlyMessage = "ارتباط سرور با موتور هوش مصنوعی برقرار نشد. سرویس پردازش تصویر موقتاً در دسترس نیست.";
        code = code || "ERR-502-CONN";
      } else if (lowerDetail.includes("timeout") || status === 504) {
        friendlyMessage = "مدت زمان پردازش سند طولانی‌تر از حد انتظار شد و درخواست با تأخیر مواجه گردید.";
        code = code || "ERR-504-TO";
      } else if (status === 400) {
        friendlyMessage = "اطلاعات یا فرمت فایل ارسالی نامعتبر است و پردازش سند امکان‌پذیر نمی‌باشد.";
        code = code || "ERR-400";
      } else {
        friendlyMessage = "متأسفانه در فرآیند استخراج متن از سند، خطایی در سامانه رخ داده است.";
        const randId = Math.random().toString(36).substring(2, 6).toUpperCase();
        code = code || `ERR-${status || 500}-${randId}`;
      }
    }

    if (!code) {
      const randId = Math.random().toString(36).substring(2, 6).toUpperCase();
      code = `ERR-${status || "SYS"}-${randId}`;
    }

    if (errorModalMessage) errorModalMessage.textContent = friendlyMessage;
    if (errorModalCode) errorModalCode.textContent = code;

    if (errorModalDetails) {
      if (detailStr && detailStr.trim()) {
        errorModalDetails.textContent = detailStr;
        errorModalDetails.classList.add("hidden");
        if (toggleErrorDetailsBtn) toggleErrorDetailsBtn.classList.remove("hidden");
        if (errorDetailsChevron) errorDetailsChevron.classList.remove("rotate-180");
      } else {
        errorModalDetails.textContent = "";
        if (toggleErrorDetailsBtn) toggleErrorDetailsBtn.classList.add("hidden");
      }
    }

    errorModal.classList.remove("hidden");
    if (window.lucide) lucide.createIcons();
  }

  function closeErrorModal() {
    if (errorModal) {
      errorModal.classList.add("hidden");
    }
  }

  if (closeErrorModalBtn) closeErrorModalBtn.addEventListener("click", closeErrorModal);
  if (errorModalCloseBtn) errorModalCloseBtn.addEventListener("click", closeErrorModal);

  if (errorModal) {
    errorModal.addEventListener("click", (e) => {
      if (e.target === errorModal) closeErrorModal();
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && errorModal && !errorModal.classList.contains("hidden")) {
      closeErrorModal();
    }
  });

  if (copyErrorCodeBtn) {
    copyErrorCodeBtn.addEventListener("click", () => {
      const codeText = errorModalCode ? errorModalCode.textContent : "";
      if (codeText) {
        navigator.clipboard.writeText(codeText).then(() => {
          if (copyErrorBtnText) {
            const original = copyErrorBtnText.textContent;
            copyErrorBtnText.textContent = "کپی شد!";
            setTimeout(() => { copyErrorBtnText.textContent = original; }, 2000);
          }
        });
      }
    });
  }

  if (toggleErrorDetailsBtn && errorModalDetails) {
    toggleErrorDetailsBtn.addEventListener("click", () => {
      const isHidden = errorModalDetails.classList.toggle("hidden");
      if (errorDetailsChevron) {
        errorDetailsChevron.classList.toggle("rotate-180", !isHidden);
      }
    });
  }

  // =========================================================================
  // Background Book Tasks Management (Submit, Poll, Progress, Retry, Download)
  // =========================================================================

  function getAuthHeaders() {
    const headers = {};
    if (currentUserToken) {
      headers["X-API-Key"] = currentUserToken;
    }
    return headers;
  }

  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function formatDurationEta(seconds) {
    if (seconds === null || seconds === undefined || seconds <= 0) return "";
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    if (mins > 0) {
      return `حدود ${mins} دقیقه و ${secs} ثانیه`;
    }
    return `حدود ${secs} ثانیه`;
  }

  // Submit Book Task (POST /api/tasks/book)
  if (submitBookTaskButton) {
    submitBookTaskButton.addEventListener("click", async () => {
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
      formData.append("cooldown_seconds", cooldownInput ? cooldownInput.value || "1.0" : "1.0");

      if (llmToggle.checked && secondaryModelSelect.value) {
        formData.append("secondary_model", secondaryModelSelect.value);
      }

      if (selectedFile.type === "application/pdf" || selectedFile.name.endsWith(".pdf")) {
        if (startPageInput.value) formData.append("start_page", startPageInput.value);
        if (endPageInput.value) formData.append("end_page", endPageInput.value);
      }

      const originalBtnHtml = submitBookTaskButton.innerHTML;
      submitBookTaskButton.disabled = true;
      submitButton.disabled = true;
      submitBookTaskButton.innerHTML = `
        <i data-lucide="loader-2" class="w-4 h-4 animate-spin text-amber-400"></i>
        <span>در حال ثبت کتاب در صف...</span>
      `;
      if (window.lucide) lucide.createIcons();

      try {
        const response = await fetch("/api/tasks/book", {
          method: "POST",
          headers: getAuthHeaders(),
          body: formData,
        });

        if (response.status === 401) {
          window.location.href = "/login";
          return;
        }

        if (!response.ok) {
          let errDetail = "خطا در ثبت وظیفه پردازش کتاب";
          try {
            const errJson = await response.json();
            errDetail = errJson.detail || errDetail;
          } catch (_) {}
          showErrorModal({ status: response.status, rawDetail: errDetail });
          return;
        }

        await fetchBookTasks();
        if (bookTasksSection) {
          window.scrollTo({ top: bookTasksSection.offsetTop - 40, behavior: "smooth" });
        }
      } catch (err) {
        showErrorModal({ status: 0, rawDetail: err.message || String(err) });
      } finally {
        submitBookTaskButton.disabled = !selectedFile;
        submitButton.disabled = !selectedFile;
        submitBookTaskButton.innerHTML = originalBtnHtml;
        if (window.lucide) lucide.createIcons();
      }
    });
  }

  if (tasksLimitSelect) {
    tasksLimitSelect.addEventListener("change", () => {
      fetchBookTasks();
    });
  }

  if (refreshTasksBtn) {
    refreshTasksBtn.addEventListener("click", () => {
      fetchBookTasks();
    });
  }

  async function fetchBookTasks() {
    if (!bookTasksList) return;
    const nVal = tasksLimitSelect ? tasksLimitSelect.value : 10;
    try {
      const res = await fetch(`/api/tasks?n=${encodeURIComponent(nVal)}`, {
        headers: getAuthHeaders(),
      });
      if (!res.ok) return;
      const data = await res.json();
      const tasks = data.tasks || [];
      renderBookTasks(tasks);

      const hasActive = tasks.some((t) => t.status === "queued" || t.status === "processing");
      if (activeBookTasksBadge) {
        activeBookTasksBadge.classList.toggle("hidden", !hasActive);
      }

      if (hasActive) {
        startTasksPolling();
      } else {
        stopTasksPolling();
      }

      if (openDetailTaskId) {
        await refreshTaskDetailModal(openDetailTaskId, false);
      }
    } catch (err) {
      console.warn("Error fetching book tasks:", err);
    }
  }

  function startTasksPolling() {
    if (tasksPollTimer) return;
    tasksPollTimer = setInterval(() => {
      fetchBookTasks();
    }, 3000);
  }

  function stopTasksPolling() {
    if (tasksPollTimer) {
      clearInterval(tasksPollTimer);
      tasksPollTimer = null;
    }
  }

  function getStatusMeta(task) {
    switch (task.status) {
      case "queued":
        return {
          label: "در صف انتظار",
          badgeClass: "bg-amber-500/20 text-amber-300 border-amber-500/30",
          barClass: "bg-amber-500",
        };
      case "processing":
        return {
          label: task.current_page ? `در حال پردازش (صفحه ${task.current_page})` : "در حال پردازش...",
          badgeClass: "bg-blue-500/20 text-blue-300 border-blue-500/30 animate-pulse",
          barClass: "bg-gradient-to-r from-blue-500 to-emerald-500",
        };
      case "completed":
        return {
          label: "تکمیل شده",
          badgeClass: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
          barClass: "bg-emerald-500",
        };
      case "completed_with_errors":
        return {
          label: `تکمیل با ${task.failed_pages} صفحه خطا`,
          badgeClass: "bg-amber-500/20 text-amber-300 border-amber-500/30",
          barClass: "bg-amber-500",
        };
      case "cancelled":
        return {
          label: "متوقف شده",
          badgeClass: "bg-zinc-700/60 text-slate-300 border-zinc-600",
          barClass: "bg-zinc-500",
        };
      default:
        return {
          label: "ناموفق",
          badgeClass: "bg-rose-500/20 text-rose-300 border-rose-500/30",
          barClass: "bg-rose-500",
        };
    }
  }

  function renderBookTasks(tasks) {
    if (!bookTasksList) return;
    if (!tasks || tasks.length === 0) {
      bookTasksList.innerHTML = `
        <div class="text-center py-8 bg-zinc-900/40 rounded-xl border border-zinc-800/80 text-xs text-slate-400 space-y-1">
          <p class="font-semibold text-slate-300">هنوز هیچ کتاب یا وظیفه پس‌زمینه‌ای ثبت نشده است.</p>
          <p class="text-[11px] text-slate-500">فایل کتاب (PDF) خود را در بخش بالا انتخاب کرده و دکمه «ثبت در صف پردازش کتاب» را بزنید.</p>
        </div>
      `;
      return;
    }

    const tokenParam = currentUserToken ? `&api_key=${encodeURIComponent(currentUserToken)}` : "";

    bookTasksList.innerHTML = tasks
      .map((t) => {
        const meta = getStatusMeta(t);
        const createdDate = t.created_at ? new Date(t.created_at).toLocaleString("fa-IR") : "";
        const pct = t.progress_percent || 0;
        const etaText = t.eta_seconds ? `زمان تقریبی باقی‌مانده: ${formatDurationEta(t.eta_seconds)}` : "";

        let downloadBtns = "";
        if (t.can_download) {
          downloadBtns = `
            <div class="flex flex-wrap items-center gap-1.5 pt-2 border-t border-zinc-800/80">
              <span class="text-[11px] text-slate-400 ml-1">دانلود خروجی کتاب:</span>
              <a href="/api/tasks/${encodeURIComponent(t.task_id)}/download?format=txt${tokenParam}" class="px-2.5 py-1 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 border border-emerald-500/30 text-emerald-300 text-[11px] font-semibold transition flex items-center space-x-1 space-x-reverse">
                <i data-lucide="file-text" class="w-3 h-3"></i>
                <span>متنی (TXT)</span>
              </a>
              <a href="/api/tasks/${encodeURIComponent(t.task_id)}/download?format=html${tokenParam}" class="px-2.5 py-1 rounded-lg bg-red-600/20 hover:bg-red-600/30 border border-red-500/30 text-red-300 text-[11px] font-semibold transition flex items-center space-x-1 space-x-reverse">
                <i data-lucide="book-open" class="w-3 h-3"></i>
                <span>کتاب چاپی (HTML)</span>
              </a>
              <a href="/api/tasks/${encodeURIComponent(t.task_id)}/download?format=md${tokenParam}" class="px-2.5 py-1 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-slate-200 text-[11px] transition">
                Markdown
              </a>
              <a href="/api/tasks/${encodeURIComponent(t.task_id)}/download?format=zip${tokenParam}" class="px-2.5 py-1 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-slate-200 text-[11px] transition">
                بسته ZIP
              </a>
              <a href="/api/tasks/${encodeURIComponent(t.task_id)}/download?format=json${tokenParam}" class="px-2.5 py-1 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-slate-200 text-[11px] transition font-mono">
                JSON
              </a>
            </div>
          `;
        }

        let errorBanner = "";
        if (t.error_message) {
          errorBanner = `
            <div class="p-2 rounded-lg bg-rose-950/40 border border-rose-500/30 text-[11px] text-rose-300 flex items-center justify-between gap-2">
              <span>${escapeHtml(t.error_message)}</span>
            </div>
          `;
        }

        return `
          <div class="p-4 rounded-xl bg-zinc-900/70 border border-zinc-800 hover:border-zinc-700 transition space-y-3">
            <div class="flex flex-wrap items-center justify-between gap-2">
              <div class="flex items-center space-x-2.5 space-x-reverse">
                <span class="px-2.5 py-0.5 rounded-full text-[11px] font-bold border ${meta.badgeClass}">
                  ${escapeHtml(meta.label)}
                </span>
                <span class="font-bold text-xs text-white">${escapeHtml(t.filename)}</span>
                <span class="text-[11px] text-red-400 bg-red-950/40 border border-red-500/20 px-2 py-0.5 rounded-md font-mono">
                  ${escapeHtml(t.primary_model)}${t.use_llm ? " + LLM" : ""}
                </span>
              </div>

              <div class="flex items-center space-x-1.5 space-x-reverse text-[11px]">
                <span class="text-slate-500 ml-2">${escapeHtml(createdDate)}</span>
                <button type="button" data-action="detail" data-task-id="${escapeHtml(t.task_id)}" class="px-2.5 py-1 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-slate-200 transition flex items-center space-x-1 space-x-reverse">
                  <i data-lucide="eye" class="w-3.5 h-3.5 text-amber-400"></i>
                  <span>جزئیات صفحات</span>
                </button>
                ${
                  t.can_retry
                    ? `<button type="button" data-action="retry" data-task-id="${escapeHtml(t.task_id)}" class="px-2.5 py-1 rounded-lg bg-amber-600/20 hover:bg-amber-600/30 border border-amber-500/30 text-amber-300 font-semibold transition flex items-center space-x-1 space-x-reverse">
                        <i data-lucide="rotate-ccw" class="w-3.5 h-3.5"></i>
                        <span>تلاش مجدد صفحات ناموفق</span>
                      </button>`
                    : ""
                }
                ${
                  t.can_cancel
                    ? `<button type="button" data-action="cancel" data-task-id="${escapeHtml(t.task_id)}" class="px-2.5 py-1 rounded-lg bg-rose-600/20 hover:bg-rose-600/30 border border-rose-500/30 text-rose-300 transition flex items-center space-x-1 space-x-reverse">
                        <i data-lucide="square" class="w-3.5 h-3.5"></i>
                        <span>توقف</span>
                      </button>`
                    : ""
                }
                <button type="button" data-action="delete" data-task-id="${escapeHtml(t.task_id)}" class="p-1.5 rounded-lg bg-zinc-800 hover:bg-rose-950/60 text-slate-400 hover:text-rose-400 transition" title="حذف وظیفه">
                  <i data-lucide="trash-2" class="w-3.5 h-3.5"></i>
                </button>
              </div>
            </div>

            <!-- Progress Bar & Page Counters -->
            <div class="space-y-1.5">
              <div class="flex flex-wrap items-center justify-between text-[11px] text-slate-400">
                <div class="flex items-center space-x-3 space-x-reverse">
                  <span>پیشرفت کل: <strong class="text-white">${pct}%</strong></span>
                  <span>موفق: <strong class="text-emerald-400">${t.completed_pages}</strong> از <strong>${t.total_pages}</strong> صفحه</span>
                  ${t.failed_pages > 0 ? `<span>ناموفق: <strong class="text-rose-400">${t.failed_pages}</strong></span>` : ""}
                  <span>استراحت بین صفحات: <strong class="text-slate-300">${t.cooldown_seconds} ثانیه</strong></span>
                </div>
                <span class="text-amber-300/90">${escapeHtml(etaText)}</span>
              </div>
              <div class="w-full h-2.5 bg-zinc-800 rounded-full overflow-hidden">
                <div class="h-full ${meta.barClass} transition-all duration-500 rounded-full" style="width: ${pct}%"></div>
              </div>
            </div>

            ${errorBanner}
            ${downloadBtns}
          </div>
        `;
      })
      .join("");

    if (window.lucide) lucide.createIcons();
  }

  // Event delegation for task action buttons (detail, retry, cancel, delete)
  if (bookTasksList) {
    bookTasksList.addEventListener("click", async (e) => {
      const btn = e.target.closest("button[data-action]");
      if (!btn) return;
      const action = btn.getAttribute("data-action");
      const taskId = btn.getAttribute("data-task-id");
      if (!taskId) return;

      if (action === "detail") {
        await openTaskDetailModal(taskId);
      } else if (action === "retry") {
        await retryBookTask(taskId);
      } else if (action === "cancel") {
        await cancelBookTask(taskId);
      } else if (action === "delete") {
        await deleteBookTask(taskId);
      }
    });
  }

  async function retryBookTask(taskId) {
    try {
      const res = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/retry`, {
        method: "POST",
        headers: getAuthHeaders(),
      });
      const data = await res.json();
      if (!res.ok) {
        showErrorModal({ status: res.status, rawDetail: data.detail || "خطا در تلاش مجدد" });
        return;
      }
      await fetchBookTasks();
    } catch (err) {
      showErrorModal({ status: 0, rawDetail: err.message });
    }
  }

  async function cancelBookTask(taskId) {
    try {
      const res = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/cancel`, {
        method: "POST",
        headers: getAuthHeaders(),
      });
      if (res.ok) {
        await fetchBookTasks();
      }
    } catch (err) {
      console.error("Failed to cancel task:", err);
    }
  }

  async function deleteBookTask(taskId) {
    if (!confirm("آیا از حذف این وظیفه و فایل کتاب مربوط به آن اطمینان دارید؟")) return;
    try {
      const res = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`, {
        method: "DELETE",
        headers: getAuthHeaders(),
      });
      if (res.ok) {
        await fetchBookTasks();
      }
    } catch (err) {
      console.error("Failed to delete task:", err);
    }
  }

  async function openTaskDetailModal(taskId) {
    if (!taskDetailModal) return;
    openDetailTaskId = taskId;
    taskDetailModal.classList.remove("hidden");
    taskDetailPagesBody.innerHTML = `<div class="text-center py-8 text-slate-400">در حال بارگذاری وضعیت صفحات...</div>`;
    await refreshTaskDetailModal(taskId, true);
  }

  async function refreshTaskDetailModal(taskId, showErrors) {
    try {
      const res = await fetch(`/api/tasks/${encodeURIComponent(taskId)}?include_pages=true`, {
        headers: getAuthHeaders(),
      });
      if (!res.ok) return;
      const t = await res.json();

      if (taskDetailTitle) {
        taskDetailTitle.textContent = `جزئیات صفحات: ${t.filename}`;
      }
      if (taskDetailSubtitle) {
        taskDetailSubtitle.textContent = `مدل: ${t.primary_model} | پیشرفت: ${t.progress_percent}% (${t.completed_pages} موفق از ${t.total_pages} صفحه)`;
      }
      if (taskDetailFooterSummary) {
        taskDetailFooterSummary.textContent = `مجموع زمان استخراج: ${t.total_ocr_duration} ثانیه | خطاها: ${t.failed_pages} صفحه`;
      }

      const pages = t.pages || [];
      taskDetailPagesBody.innerHTML = pages
        .map((p) => {
          let statusBadge = "";
          if (p.status === "completed") {
            statusBadge = `<span class="px-2 py-0.5 rounded-md bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-[10px] font-bold">تکمیل شده (${p.ocr_duration}s)</span>`;
          } else if (p.status === "processing") {
            statusBadge = `<span class="px-2 py-0.5 rounded-md bg-blue-500/20 text-blue-300 border border-blue-500/30 text-[10px] font-bold animate-pulse">در حال استخراج...</span>`;
          } else if (p.status === "failed") {
            statusBadge = `<span class="px-2 py-0.5 rounded-md bg-rose-500/20 text-rose-300 border border-rose-500/30 text-[10px] font-bold">ناموفق (${p.retry_count} تلاش)</span>`;
          } else {
            statusBadge = `<span class="px-2 py-0.5 rounded-md bg-zinc-800 text-slate-400 text-[10px]">در نوبت</span>`;
          }

          return `
            <div class="p-3 rounded-xl bg-zinc-900/90 border border-zinc-800 space-y-2">
              <div class="flex items-center justify-between">
                <div class="flex items-center space-x-2 space-x-reverse">
                  <span class="font-bold text-white">صفحه ${p.page_number}</span>
                  ${statusBadge}
                  ${p.retry_count > 0 && p.status === "completed" ? `<span class="text-[10px] text-amber-400">(موفق پس از ${p.retry_count} تلاش مجدد)</span>` : ""}
                </div>
                <span class="text-[10px] text-slate-500">${p.char_count || 0} کاراکتر</span>
              </div>
              ${
                p.last_error
                  ? `<div class="p-2 rounded-lg bg-rose-950/40 border border-rose-500/30 text-[11px] text-rose-300">${escapeHtml(p.last_error)}</div>`
                  : ""
              }
              ${
                p.text
                  ? `<div class="p-2.5 rounded-lg bg-black/50 border border-zinc-800/80 text-slate-200 text-xs leading-relaxed max-h-36 overflow-y-auto whitespace-pre-wrap">${escapeHtml(p.text)}</div>`
                  : ""
              }
            </div>
          `;
        })
        .join("");

      if (window.lucide) lucide.createIcons();
    } catch (err) {
      if (showErrors) {
        console.error("Failed to load task details:", err);
      }
    }
  }

  function closeTaskDetailModal() {
    openDetailTaskId = null;
    if (taskDetailModal) {
      taskDetailModal.classList.add("hidden");
    }
  }

  if (closeTaskDetailBtn) closeTaskDetailBtn.addEventListener("click", closeTaskDetailModal);
  if (taskDetailCloseFooterBtn) taskDetailCloseFooterBtn.addEventListener("click", closeTaskDetailModal);
  if (taskDetailModal) {
    taskDetailModal.addEventListener("click", (e) => {
      if (e.target === taskDetailModal) closeTaskDetailModal();
    });
  }
});
