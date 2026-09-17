'use strict';
const $ = id => document.getElementById(id);
const form = $('form');
let busy = false;
let hasResult = false;
let timer = null;
const videoExtensions = /\.(mp4|avi|mov|mkv|webm)$/i;
const imageExtensions = /\.(png|jpe?g|webp|bmp)$/i;
const isVlm = () => $('backend').value === 'vlm';

function showError(message) {
  $('form-error').textContent = message;
  $('form-error').hidden = false;
}
function clearError() {
  $('form-error').hidden = true;
  $('form-error').textContent = '';
}
function syncMode() {
  $('vlm-fields').hidden = !isVlm();
  $('vlm-fields').disabled = !isVlm();
  $('backend-note').textContent = isVlm()
    ? '结合完整视频与参考图理解场景。需先在服务端配置模型。'
    : '基于图像匹配，无需模型服务。适合验证参考视角匹配流程。';
}
function updateFileLabel() {
  const file = $('video').files[0];
  $('file-title').textContent = file ? file.name : '选择视频，或拖到这里';
  $('file-detail').textContent = file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · 点击更换视频` : 'MP4 / AVI / MOV / MKV / WebM';
  if (!hasResult) {
    $('empty-title').textContent = file ? '视频已就位' : '好画面，从这里开始';
    $('empty-description').textContent = file ? '确认左侧目标和分析方式，然后开始分析。' : '添加一段视频，建立当前画面与目标之间的联系。';
    $('stage-status').textContent = file ? '已选视频' : '等待输入';
  }
}
form.addEventListener('change', () => {
  syncMode(); updateFileLabel(); clearError();
  if (hasResult) $('stale-note').hidden = false;
});
form.addEventListener('input', () => {
  if (hasResult && !busy) $('stale-note').hidden = false;
});
for (const name of ['dragenter', 'dragover']) $('dropzone').addEventListener(name, event => {
  event.preventDefault();
  if (!busy) $('dropzone').classList.add('dragover');
});
for (const name of ['dragleave', 'drop']) $('dropzone').addEventListener(name, () => $('dropzone').classList.remove('dragover'));
$('dropzone').addEventListener('drop', event => {
  event.preventDefault();
  if (busy) return;
  if (event.dataTransfer.files.length !== 1) { showError('每次请选择一个视频文件。'); return; }
  const file = event.dataTransfer.files[0];
  if (!videoExtensions.test(file.name)) { showError('请选择 MP4、AVI、MOV、MKV 或 WebM 视频。'); return; }
  const transfer = new DataTransfer(); transfer.items.add(file);
  $('video').files = transfer.files;
  $('video').dispatchEvent(new Event('change', {bubbles: true}));
});

function validateInputs() {
  const file = $('video').files[0];
  if (!file) return ['请先选择一段视频。', 'video'];
  if (!videoExtensions.test(file.name)) return ['视频格式不支持，请选择 MP4、AVI、MOV、MKV 或 WebM。', 'video'];
  if (!file.size) return ['视频文件为空，请重新选择。', 'video'];
  if (file.size > 200 * 1024 * 1024) return ['视频不能超过 200 MB。', 'video'];
  for (const [id, name] of [['current-frame', '当前帧'], ['reference-frame', 'Reference Frame'], ['target-sketch', 'Target Sketch']]) {
    const image = $(id).files[0];
    if (!image || !imageExtensions.test(image.name) || !image.size || image.size > 20 * 1024 * 1024) return [`请上传${name}图片，且不超过 20 MB。`, id];
  }
  for (const id of ['target-state', 'render-meta']) {
    try { const value = JSON.parse($(id).value); if (!value || Array.isArray(value) || typeof value !== 'object') throw new Error(); }
    catch { return ['请填写有效的 JSON 对象。', id]; }
  }
  if (isVlm()) {
    const url = $('video_url').value.trim();
    if (url) {
      try { const parsed = new URL(url); if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password) throw new Error(); }
      catch { return ['原视频 URL 必须是有效的 HTTP 或 HTTPS 地址。', 'video_url']; }
    }
    if (!url && 4 * Math.ceil(file.size / 3) >= 100000000) return ['视频编码后超过当前服务 100 MB 上限，请填写同一原视频的公网 URL。', 'video_url'];
  }
  return null;
}
function artifactUrl(value) {
  if (typeof value !== 'string' || !/^\/api\/runs\/[a-f0-9]{32}\//.test(value)) throw new Error('服务返回的结果文件地址无效。');
  const url = new URL(value, location.origin);
  if (url.origin !== location.origin || !url.pathname.startsWith('/api/runs/')) throw new Error('结果文件地址无效。');
  return url.href;
}
function validateResponse(body) {
  const r = body?.result;
  if (!r || !['navigation', 'composition', 'reached', 'uncertain'].includes(r.phase) || !Array.isArray(r.actions) || !Number.isFinite(r.confidence) || typeof r.guidance !== 'string' || !Array.isArray(r.warnings)) throw new Error('服务返回的分析结果不完整。');
  for (const key of ['current_frame', 'reference_frame', 'target_sketch', 'result']) artifactUrl(body.files?.[key]);
}
function imageCard(url, title, subtitle) {
  const figure = document.createElement('figure'); figure.className = 'image-card';
  const button = document.createElement('button'); button.type = 'button'; button.className = 'image-button'; button.setAttribute('aria-label', `放大查看${title}`);
  const image = document.createElement('img'); image.src = artifactUrl(url); image.alt = title;
  image.addEventListener('error', () => { const error = document.createElement('span'); error.className = 'image-error'; error.textContent = '图片加载失败'; button.replaceChildren(error); button.disabled = true; });
  button.append(image);
  button.addEventListener('click', () => { $('dialog-image').src = image.src; $('dialog-image').alt = title; $('dialog-caption').textContent = title; $('image-dialog').showModal(); });
  const caption = document.createElement('figcaption'); caption.textContent = title;
  const label = document.createElement('span'); label.textContent = subtitle; caption.append(label);
  figure.append(button, caption); return figure;
}
function renderResult(body, snapshot) {
  const r = body.result;
  $('result-kicker').textContent = r.phase.toUpperCase();
  $('result-title').textContent = {navigation: '先接近参考视角', composition: '调整至目标构图', reached: '已接近目标画面', uncertain: '需要进一步判断'}[r.phase];
  $('result-description').textContent = r.guidance;
  $('score').textContent = `${Math.round(r.confidence * 100)}%`;
  $('score').title = '后端返回的参考置信度，不是准确率或完成度。';
  $('guidance').textContent = r.guidance;
  $('camera').textContent = r.actions.map(a => `${a.actor}: ${a.action}${a.reason ? ' · ' + a.reason : ''}`).join('；') || '暂无动作';
  $('result-context').textContent = `${snapshot.backend} · ${snapshot.mode} · ${snapshot.filename}`;
  const times = body.timing || {};
  const labels = [];
  if (Number.isFinite(times.total_seconds)) labels.push(`总耗时 ${times.total_seconds.toFixed(2)} 秒`);
  if (Number.isFinite(times.vlm_seconds)) labels.push(`模型 ${times.vlm_seconds.toFixed(2)} 秒`);
  $('timing').textContent = labels.join(' / ');
  $('compare-grid').replaceChildren(...[['current_frame', '当前帧', 'CURRENT'], ['reference_frame', '参考帧', 'REFERENCE'], ['target_sketch', '目标画面', 'TARGET SKETCH']].map(([key, title, subtitle]) => imageCard(body.files[key], title, subtitle)));
  $('warnings').replaceChildren(...r.warnings.map(text => { const li = document.createElement('li'); li.textContent = text; return li; }));
  $('warnings-box').hidden = !r.warnings.length;
  $('task-label').textContent = `任务 ${body.task_id.slice(0, 8)}`; $('task-label').title = body.task_id;
  $('download').href = artifactUrl(body.files.result);
  $('result').hidden = false; $('empty').hidden = true; $('stale-note').hidden = true;
  $('stage-status').textContent = r.phase;
  hasResult = true;
}
form.addEventListener('submit', async event => {
  event.preventDefault(); if (busy) return;
  clearError(); const invalid = validateInputs();
  if (invalid) { showError(invalid[0]); $(invalid[1]).focus(); return; }
  const snapshot = {filename: $('video').files[0].name, backend: isVlm() ? 'Qwen 视频理解' : '本地视觉', mode: 'PhotoAgent 目标'};
  const data = new FormData(form);
  if (isVlm()) data.set('video_url', $('video_url').value.trim()); else data.delete('video_url');
  busy = true; $('controls').disabled = true; $('submit').disabled = true; $('submit-label').textContent = '正在分析'; $('stage').setAttribute('aria-busy', 'true'); $('activity').hidden = false; $('stage-status').textContent = '分析中';
  if (hasResult) $('stale-note').hidden = false;
  const started = Date.now(); $('elapsed').textContent = '已等待 0 秒';
  timer = setInterval(() => $('elapsed').textContent = `已等待 ${Math.floor((Date.now() - started) / 1000)} 秒`, 1000);
  try {
    const response = await fetch('/api/analyze', {method: 'POST', body: data});
    let body;
    try { body = await response.json(); } catch { throw new Error(`服务响应无法解析（HTTP ${response.status}），请确认后端服务正在运行。`); }
    if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `提交失败（HTTP ${response.status}），请检查输入文件。`);
    validateResponse(body); renderResult(body, snapshot);
  } catch (error) {
    showError((error instanceof TypeError ? '无法连接分析服务，请检查服务是否运行。请求可能仍在服务端执行。' : error.message) + (hasResult ? ' 下方保留上一次成功的结果。' : ''));
    $('stage-status').textContent = '本次未完成';
  } finally {
    clearInterval(timer); timer = null; busy = false; $('controls').disabled = false; syncMode(); $('submit').disabled = false; $('submit-label').textContent = hasResult ? '重新分析' : '开始分析'; $('stage').setAttribute('aria-busy', 'false'); $('activity').hidden = true;
  }
});
$('close-dialog').addEventListener('click', () => $('image-dialog').close());
$('image-dialog').addEventListener('click', event => { if (event.target === $('image-dialog')) { const rect = $('image-dialog').getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $('image-dialog').close(); } });
window.addEventListener('beforeunload', event => { if (busy) { event.preventDefault(); event.returnValue = ''; } });
syncMode(); updateFileLabel();
