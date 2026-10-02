const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value).replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));

const conversation = $('#conversation');
const queryForm = $('#query-form');
const queryInput = $('#query-input');
const roleSelect = $('#role-select');
const traceState = $('#trace-state');
const traceSteps = $('#trace-steps');
const citationList = $('#citation-list');
const citationCount = $('#citation-count');
const panels = document.querySelectorAll('.panel');
const navItems = document.querySelectorAll('.nav-item');
const uploadButtons = [$('#open-upload'), $('#library-upload')];
const dialog = $('#upload-dialog');
const closeDialogBtn = $('#close-upload-dialog');
const fileInput = $('#file-input');
const fileDropLabel = $('#file-drop-label');
const uploadForm = $('#upload-form');
const uploadFeedback = $('#upload-feedback');

// Institutional roles allowed to manage the knowledge base
const managingRoles = new Set(['administrator', 'staff', 'faculty']);

function canManageKnowledgeBase() {
  return managingRoles.has(roleSelect.value);
}

function updateUploadAccess() {
  const allowed = canManageKnowledgeBase();
  const currentRole = roleSelect.value;
  const capitalizedRole = currentRole.charAt(0).toUpperCase() + currentRole.slice(1);

  uploadButtons.forEach((button) => {
    if (!button) return;
    button.disabled = !allowed;
    button.setAttribute('aria-disabled', String(!allowed));
    if (allowed) {
      button.title = `Add material to the knowledge base (${capitalizedRole})`;
      button.classList.remove('restricted');
    } else {
      button.title = 'Only administrator, staff and faculty are able to add the file to the knowledge base';
      button.classList.add('restricted');
    }
  });

  const note = $('#upload-permission-note');
  if (note) {
    if (allowed) {
      note.className = 'upload-permission-note';
      note.innerHTML = `<div class="perm-banner allowed"><span class="perm-icon">✓</span> <span><strong>Authorized (${escapeHtml(capitalizedRole)}):</strong> You can add material to the shared knowledge base. All uploaded files persist across reloads.</span></div>`;
    } else {
      note.className = 'upload-permission-note';
      note.innerHTML = `<div class="perm-banner restricted"><span class="perm-icon">🔒</span> <span><strong>Upload restricted:</strong> Only administrator, staff and faculty are able to add the file to the knowledge base. Switch your access role in Session settings to add material.</span></div>`;
    }
  }

  const uploaderRoleInfo = $('#uploader-role-info');
  if (uploaderRoleInfo) {
    uploaderRoleInfo.innerHTML = `Role authorized for upload: <strong>${escapeHtml(capitalizedRole)}</strong>`;
  }
}

// Restore saved role from localStorage if available
const savedRole = localStorage.getItem('academia_user_role');
if (savedRole && ['student', 'faculty', 'staff', 'administrator'].includes(savedRole)) {
  roleSelect.value = savedRole;
}

roleSelect.addEventListener('change', () => {
  localStorage.setItem('academia_user_role', roleSelect.value);
  updateUploadAccess();
});

function setPanel(name) {
  panels.forEach((panel) => panel.classList.toggle('active', panel.id === `${name}-panel`));
  navItems.forEach((item) => item.classList.toggle('active', item.dataset.panel === name));
  localStorage.setItem('academia_active_panel', name);
  if (name === 'library') loadDocuments();
}

navItems.forEach((item) => item.addEventListener('click', () => setPanel(item.dataset.panel)));

// Restore saved panel from localStorage if available
const savedPanel = localStorage.getItem('academia_active_panel');
if (savedPanel && ['chat', 'audit', 'library'].includes(savedPanel)) {
  setPanel(savedPanel);
}

function appendMessage(kind, content, extra = '') {
  const message = document.createElement('article');
  message.className = `message ${kind}-message`;
  message.innerHTML = kind === 'assistant'
    ? `<div class="avatar">A</div><div class="message-body">${content}${extra}</div>`
    : `<div class="message-body">${escapeHtml(content)}</div>`;
  conversation.append(message);
  conversation.scrollTop = conversation.scrollHeight;
  return message;
}

function setLoading(isLoading) {
  queryForm.classList.toggle('loading', isLoading);
  queryInput.disabled = isLoading;
  queryForm.querySelector('button').disabled = isLoading;
}

function showTrace(data, route) {
  traceState.textContent = data.retrieval_quality === 'grounded' ? 'Grounded' : data.retrieval_quality;
  traceState.className = `trace-state ${data.retrieval_quality}`;
  traceSteps.innerHTML = `
    <li class="complete"><span class="step-index">01</span><div><strong>${escapeHtml(route)} agent</strong><small>${escapeHtml(data.notes[0] || 'Intent routed')}</small></div></li>
    <li class="complete"><span class="step-index">02</span><div><strong>Hybrid retrieval · α ${data.hybrid_alpha.toFixed(2)}</strong><small>${data.loop_count} pass${data.loop_count === 1 ? '' : 'es'} · ${escapeHtml(data.effective_query)}</small></div></li>
    <li class="complete"><span class="step-index">03</span><div><strong>Reflection</strong><small>${escapeHtml(data.notes.at(-1) || 'Completed')}</small></div></li>`;
}

function renderCitations(citations = []) {
  citationCount.textContent = citations.length;
  citationList.innerHTML = citations.length ? citations.map((citation, index) => `
    <article class="citation"><div class="citation-top"><span>[${index + 1}]</span><strong>${escapeHtml(citation.title)}</strong><span>${Math.round(citation.score * 100)}%</span></div><small>${escapeHtml(citation.section)}</small><p>${escapeHtml(citation.excerpt)}</p></article>`).join('') : '<p class="muted">No source was used for this response.</p>';
}

function renderResponse(response) {
  const answer = escapeHtml(response.answer).replace(/\n/g, '<br>');
  const followUps = response.follow_up_questions?.length ? `<div class="follow-ups">${response.follow_up_questions.map((question) => `<button type="button">${escapeHtml(question)}</button>`).join('')}</div>` : '';
  appendMessage('assistant', `<div class="message-meta"><span>${escapeHtml(response.route)} agent</span><span>grounded response</span></div><div class="answer">${answer}</div>${followUps}`);
  document.querySelectorAll('.follow-ups button').forEach((button) => button.addEventListener('click', () => {
    queryInput.value = button.textContent;
    queryInput.focus();
  }));
  showTrace(response.trace, response.route);
  renderCitations(response.citations);
}

queryForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) return;
  appendMessage('user', query);
  queryInput.value = '';
  setLoading(true);
  traceState.textContent = 'Thinking';
  try {
    const response = await fetch('/api/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query, role: roleSelect.value }) });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'The agent could not process that question.');
    renderResponse(body);
  } catch (error) {
    appendMessage('assistant', `<div class="answer error">${escapeHtml(error.message)}</div>`);
    traceState.textContent = 'Error';
  } finally { setLoading(false); queryInput.focus(); }
});

queryInput.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') queryForm.requestSubmit();
});
document.querySelectorAll('[data-prompt]').forEach((button) => button.addEventListener('click', () => { queryInput.value = button.dataset.prompt; queryForm.requestSubmit(); }));

async function loadDocuments() {
  const target = $('#document-list');
  try {
    const response = await fetch('/api/documents', { cache: 'no-store' });
    const documents = await response.json();
    if (!Array.isArray(documents) || documents.length === 0) {
      target.innerHTML = '<div class="empty-state">No documents are available in the knowledge base yet.</div>';
      return;
    }
    target.innerHTML = documents.map((document) => {
      const isUpload = document.source && document.source.startsWith('upload:');
      const badge = isUpload
        ? '<span class="source-tag upload-tag">Uploaded Source</span>'
        : '<span class="source-tag seed-tag">Core Seed</span>';
      return `
        <article class="document-card">
          <div class="document-icon">${isUpload ? '📄' : '▤'}</div>
          <div class="document-card-content">
            <div class="document-card-header">
              <h3>${escapeHtml(document.title)}</h3>
              ${badge}
            </div>
            <p>${document.chunks} retrieval chunk${document.chunks === 1 ? '' : 's'} · <span class="doc-source">${escapeHtml(document.source)}</span></p>
            <div class="role-chips">${document.allowed_roles.map((role) => `<span>${escapeHtml(role)}</span>`).join('')}</div>
          </div>
        </article>`;
    }).join('');
  } catch (err) {
    target.innerHTML = '<div class="empty-state">Could not load the knowledge base.</div>';
  }
}

function openUpload() {
  if (!canManageKnowledgeBase()) {
    alert('Only administrator, staff and faculty are able to add the file to the knowledge base.');
    return;
  }
  uploadFeedback.textContent = '';
  uploadFeedback.className = 'upload-feedback';
  if (fileDropLabel) fileDropLabel.textContent = 'Choose a source file';
  if (fileInput) fileInput.value = '';
  updateUploadAccess();
  dialog.showModal();
}

if ($('#open-upload')) $('#open-upload').addEventListener('click', openUpload);
if ($('#library-upload')) $('#library-upload').addEventListener('click', openUpload);

if (closeDialogBtn) {
  closeDialogBtn.addEventListener('click', () => {
    dialog.close();
  });
}

if (fileInput && fileDropLabel) {
  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) {
      const file = fileInput.files[0];
      const sizeKb = Math.round(file.size / 1024);
      fileDropLabel.textContent = `${file.name} (${sizeKb} KB)`;
    } else {
      fileDropLabel.textContent = 'Choose a source file';
    }
  });
}

uploadForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!canManageKnowledgeBase()) {
    uploadFeedback.className = 'upload-feedback error';
    uploadFeedback.textContent = 'Only administrator, staff and faculty are able to add the file to the knowledge base.';
    return;
  }
  if (!fileInput.files || !fileInput.files[0]) {
    uploadFeedback.className = 'upload-feedback error';
    uploadFeedback.textContent = 'Please choose a file to upload.';
    return;
  }
  const selectedRoles = [...$('#upload-roles').selectedOptions].map((option) => option.value).join(',');
  const form = new FormData();
  form.append('file', fileInput.files[0]);
  form.append('roles', selectedRoles);
  form.append('uploader_role', roleSelect.value);

  uploadFeedback.className = 'upload-feedback';
  uploadFeedback.textContent = 'Chunking, extracting, and persisting to knowledge base…';
  const submitBtn = uploadForm.querySelector('.dialog-submit');
  if (submitBtn) submitBtn.disabled = true;

  try {
    const response = await fetch('/api/documents', { method: 'POST', body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Upload failed.');
    uploadFeedback.className = 'upload-feedback success';
    uploadFeedback.textContent = `"${body.title}" successfully added and persisted to the knowledge base.`;
    fileInput.value = '';
    if (fileDropLabel) fileDropLabel.textContent = 'Choose a source file';
    await loadDocuments();
    fetchHealth();
    setTimeout(() => {
      dialog.close();
      uploadFeedback.textContent = '';
    }, 1800);
  } catch (error) {
    uploadFeedback.className = 'upload-feedback error';
    uploadFeedback.textContent = error.message;
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
});

updateUploadAccess();

$('#audit-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const target = $('#audit-result');
  target.className = 'audit-result loading-result'; target.textContent = 'Checking requirements…';
  const completed_courses = $('#courses-input').value.split(',').map((course) => course.trim()).filter(Boolean);
  try {
    const response = await fetch('/api/degree-audit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ completed_courses, earned_credits: Number($('#credits-input').value), cumulative_gpa: Number($('#gpa-input').value) }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'The audit could not run.');
    target.className = 'audit-result';
    target.innerHTML = `<div class="audit-status ${result.status.replace(' ', '-')}"><span>${result.status === 'on track' ? '✓' : '!'}</span>${escapeHtml(result.status)}</div><div class="audit-columns"><div><h3>Eligible next steps</h3><ul>${result.eligible_courses.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul></div><div><h3>Requirements to review</h3><ul>${result.outstanding_requirements.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul></div></div><p class="advisory">${escapeHtml(result.advisory)}</p>`;
    renderCitations(result.citations);
  } catch (error) { target.className = 'audit-result error'; target.textContent = error.message; }
});

function fetchHealth() {
  fetch('/api/health')
    .then((response) => response.json())
    .then((health) => {
      $('#health-label').textContent = `${health.documents} sources · ${health.chunks} chunks online`;
    })
    .catch(() => {
      $('#health-label').textContent = 'Offline';
    });
}

fetchHealth();
loadDocuments();
