(() => {
  'use strict';
  const form = document.getElementById('uploadForm');
  const input = document.getElementById('fileInput');
  const drop = document.getElementById('dropzone');
  const selected = document.getElementById('selectedFile');
  const preview = document.getElementById('localPreview');
  const error = document.getElementById('errorMessage');
  const button = document.getElementById('analyzeButton');
  const processing = document.getElementById('processing');
  const allowed = /\.(mp4|mov|avi|mkv|webm|m4v)$/i;
  let previewURL = null;
  let busy = false;
  document.getElementById('faceLayout').addEventListener('change', e => {
    if (e.target.value === 'repeated_3x3') form.elements.include_faces.checked = true;
  });
  function showError(message) { error.textContent = message; error.hidden = !message; }
  function choose() {
    const file = input.files[0];
    if (!file) return;
    if (!allowed.test(file.name) || file.size === 0 || file.size > 500 * 1024 * 1024) {
      showError('Choose a supported, nonempty video smaller than 500 MB.');
      input.value = ''; selected.hidden = true; drop.hidden = false;
      preview.removeAttribute('src'); preview.load();
      if (previewURL) URL.revokeObjectURL(previewURL);
      previewURL = null; return;
    }
    showError('');
    if (previewURL) URL.revokeObjectURL(previewURL);
    previewURL = URL.createObjectURL(file);
    preview.src = previewURL;
    document.getElementById('selectedName').textContent = file.name;
    document.getElementById('selectedSize').textContent = (file.size / 1048576).toFixed(2) + ' MB · Ready to analyze';
    document.getElementById('previewNote').hidden = true;
    selected.hidden = false; drop.hidden = true;
  }
  preview.addEventListener('error', () => { document.getElementById('previewNote').hidden = false; });
  input.addEventListener('change', choose);
  document.getElementById('changeFile').addEventListener('click', () => { if (!busy) input.click(); });
  drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  ['dragenter', 'dragover'].forEach(name => drop.addEventListener(name, e => { e.preventDefault(); drop.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach(name => drop.addEventListener(name, e => { e.preventDefault(); drop.classList.remove('dragging'); }));
  drop.addEventListener('drop', e => {
    if (!e.dataTransfer.files.length || busy) return;
    const transfer = new DataTransfer(); transfer.items.add(e.dataTransfer.files[0]);
    input.files = transfer.files; choose();
  });
  form.addEventListener('submit', async e => {
    e.preventDefault();
    if (busy || !input.files.length) return;
    const data = new FormData(form);
    busy = true; button.disabled = true; processing.hidden = false;
    form.setAttribute('aria-busy', 'true'); preview.pause(); showError('');
    const started = Date.now();
    const timer = setInterval(() => { document.getElementById('elapsed').textContent = Math.floor((Date.now() - started) / 1000); }, 1000);
    try {
      const response = await fetch(form.action, { method: 'POST', body: data, headers: { Accept: 'application/json' } });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Analysis could not finish. Please try again.');
      if (!result.redirect) throw new Error('The server did not return a result page.');
      window.location.assign(result.redirect);
    } catch (err) {
      showError(err instanceof TypeError ? 'Could not reach the Flask server. Check that it is running and try again.' : err.message);
      busy = false; button.disabled = false; processing.hidden = true;
      form.removeAttribute('aria-busy');
    } finally { clearInterval(timer); }
  });
  window.addEventListener('pagehide', () => { if (previewURL) URL.revokeObjectURL(previewURL); });
})();
