const $ = (id) => document.getElementById(id);
let selectedJobId = null;
let selectedLogOffset = 0;
let refreshBusy = false;

const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmtBytes = (n) => n < 1024 ? `${n} B` : n < 1024*1024 ? `${(n/1024).toFixed(1)} KB` : `${(n/1024/1024).toFixed(1)} MB`;
const fmtTime = (seconds) => seconds ? new Date(seconds*1000).toLocaleString() : '—';

async function api(path, options={}) {
  const response = await fetch(path, {headers:{'Content-Type':'application/json', ...(options.headers||{})}, ...options});
  const data = await response.json().catch(()=>({detail:response.statusText}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}
function showMessage(id, text, kind='') { const el=$(id); el.textContent=text; el.className=`form-message ${kind}`.trim(); }
function currentKind() { return $('job-kind').value; }
function toggleFields() {
  const kind=currentKind();
  for (const name of ['rl','synthetic','evaluate','world-model']) $(`fields-${name}`).classList.toggle('hidden', name!==kind);
  $('start-button').textContent = ({rl:'Start training job',synthetic:'Collect transitions',evaluate:'Run evaluation', 'world-model':'Train world model'})[kind];
}
$('job-kind').addEventListener('change', toggleFields);
$('algorithm')?.addEventListener('change', () => {
  const robot=$('robot-name')?.value || 'panda';
  const value=$('rl-output').value;
  if (value === 'training/runs/panda_ppo' || value === `training/runs/${robot}_ppo` || value === `training/runs/${robot}_sac`) {
    $('rl-output').value=`training/runs/${robot}_${$('algorithm').value.toLowerCase()}`;
  }
});

function buildJobPayload() {
  switch(currentKind()) {
    case 'rl': return ['/api/jobs/rl-train', {robot_name:$('robot-name').value,algorithm:$('algorithm').value,timesteps:Number($('timesteps').value),seed:Number($('seed').value),device:$('device').value,output:$('rl-output').value.trim()}];
    case 'synthetic': return ['/api/jobs/synthetic-collect',{robot_name:$('robot-name').value,episodes:Number($('episodes').value),seed:Number($('synthetic-seed').value),output:$('synthetic-output').value.trim()}];
    case 'evaluate': return ['/api/jobs/evaluate',{robot_name:$('robot-name').value,checkpoint:$('eval-checkpoint').value.trim(),algorithm:$('eval-algorithm').value,episodes:Number($('eval-episodes').value),seed:Number($('eval-seed').value)}];
    default: return ['/api/jobs/world-model-train',{dataset:$('wm-dataset').value.trim(),output:$('wm-output').value.trim(),epochs:Number($('wm-epochs').value)}];
  }
}
$('job-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button=$('start-button'); button.disabled=true; showMessage('form-message','Submitting job…');
  try {
    const [url,payload]=buildJobPayload(); const job=await api(url,{method:'POST',body:JSON.stringify(payload)});
    selectedJobId=job.id; selectedLogOffset=0; showMessage('form-message',`Job ${job.id} accepted. Process startup and completion are tracked separately.`, 'success');
    await refreshAll(); await selectJob(job.id);
  } catch(error) { showMessage('form-message',error.message,'error'); }
  finally {button.disabled=false;}
});

function renderJobs(jobs) {
  const root=$('jobs-list');
  if (!jobs.length) {root.innerHTML='<div class="empty-state">No jobs submitted in this service session. Launch an experiment to begin.</div>';return;}
  root.innerHTML=jobs.map(job=>`<button class="job-item ${job.id===selectedJobId?'selected':''}" data-job-id="${escapeHtml(job.id)}"><span class="job-info"><span class="job-name">${escapeHtml(({rl_train:'RL policy training',synthetic_collect:'Synthetic transition collection',evaluate:'Policy evaluation',world_model_train:'World-model training'})[job.kind]||job.kind)}</span><span class="job-details">${escapeHtml(job.id)} · ${escapeHtml(fmtTime(job.created_at))}</span><span class="job-details">${escapeHtml(job.command?.slice(1,4).join(' ')||'')}</span></span><span class="job-right"><span class="job-state ${escapeHtml(job.status)}">${escapeHtml(job.status)}</span><span class="job-details">${job.log_count||0} log lines</span></span></button>`).join('');
  root.querySelectorAll('[data-job-id]').forEach(button=>button.addEventListener('click',()=>selectJob(button.dataset.jobId)));
}
function renderFiles(targetId, items, emptyText) {
  const root=$(targetId);
  if (!items.length) {root.innerHTML=`<div class="empty-state">${escapeHtml(emptyText)}</div>`;return;}
  root.innerHTML=items.map(item=>`<div class="file-row"><div><div class="file-name">${escapeHtml(item.name)}</div><div class="file-meta">${escapeHtml(item.path)} · ${fmtBytes(item.size_bytes)}</div></div><a href="/api/artifacts/download?path=${encodeURIComponent(item.path)}">Download</a></div>`).join('');
}
async function selectJob(jobId) {
  selectedJobId=jobId; selectedLogOffset=0; $('job-logs').textContent='Loading logs…'; $('stop-job-button').disabled=true;
  try { const job=await api(`/api/jobs/${encodeURIComponent(jobId)}`); $('logs-title').textContent=`Job logs · ${job.kind}`; $('selected-job-meta').textContent=`${job.status.toUpperCase()} · started ${fmtTime(job.started_at)} · finished ${fmtTime(job.finished_at)} · exit code ${job.exit_code ?? '—'}${job.error?` · ${job.error}`:''}`; $('job-logs').textContent=(job.logs||[]).join('\n')||'No output yet. The process may still be starting.'; $('job-logs').scrollTop=$('job-logs').scrollHeight; $('stop-job-button').disabled=!['queued','starting','running'].includes(job.status); }
  catch(error) {$('job-logs').textContent=error.message;}
  renderJobs((await api('/api/jobs')).items);
}
$('stop-job-button').addEventListener('click', async()=>{
  if(!selectedJobId)return; $('stop-job-button').disabled=true;
  try {await api(`/api/jobs/${encodeURIComponent(selectedJobId)}/stop`,{method:'POST'}); await selectJob(selectedJobId);}
  catch(error){$('selected-job-meta').textContent=error.message;}
});
async function loadRobots() {
  const select=$('robot-name');
  if (!select) return;
  try {
    const data=await api('/api/robots');
    const previous=select.value;
    select.replaceChildren();
    for (const robot of data.items||[]) {
      const option=document.createElement('option');
      option.value=robot.id; option.textContent=robot.display_name||robot.id;
      select.append(option);
    }
    if ([...select.options].some(option=>option.value===previous)) select.value=previous;
  } catch (error) { showMessage('form-message',`Could not load robot profiles: ${error.message}`,'error'); }
}
let activeRobotProfile = 'panda';
loadRobots();
$('robot-name')?.addEventListener('change', () => {
  const robot=$('robot-name').value;
  if ($('rl-output').value === `training/runs/${activeRobotProfile}_ppo` || $('rl-output').value === 'training/runs/panda_ppo') {
    $('rl-output').value=`training/runs/${robot}_${$('algorithm').value.toLowerCase()}`;
  }
  if ($('synthetic-output').value === `training/datasets/${activeRobotProfile}_transitions.npz` || $('synthetic-output').value === 'training/datasets/transitions.npz') {
    $('synthetic-output').value=`training/datasets/${robot}_transitions.npz`;
  }
  activeRobotProfile=robot;
});

async function refreshAll() {
  if(refreshBusy)return; refreshBusy=true;
  try {
    const [health,overview,jobs,datasets,artifacts]=await Promise.all([api('/health'),api('/api/overview'),api('/api/jobs'),api('/api/datasets'),api('/api/artifacts')]);
    $('service-status').textContent=`Online · ${health.application}`; $('active-jobs').textContent=overview.active_jobs; $('total-jobs').textContent=overview.total_jobs; $('dataset-count').textContent=overview.datasets.length; $('artifact-count').textContent=artifacts.items.length;
    renderJobs(jobs.items); renderFiles('datasets-list',datasets.items,'No training datasets yet. Collect transitions to create one.'); renderFiles('artifacts-list',artifacts.items,'No artifacts found. Completed runs will appear here.');
    if(selectedJobId && jobs.items.some(j=>j.id===selectedJobId)) await selectJob(selectedJobId);
  } catch(error) {$('service-status').textContent='Offline or API error'; $('service-status').title=error.message;}
  finally {refreshBusy=false;}
}
$('refresh-button').addEventListener('click',refreshAll);
$('import-button').addEventListener('click',async()=>{
  const source=$('import-source').value.trim(),name=$('import-name').value.trim();
  if(!source||!name){showMessage('import-message','Enter both a project-relative source path and a destination filename.','error');return;}
  const button=$('import-button');button.disabled=true;
  try{const result=await api('/api/artifacts/import',{method:'POST',body:JSON.stringify({source,destination_name:name})});showMessage('import-message',`Imported to ${result.destination}.`, 'success');$('import-source').value='';await refreshAll();}
  catch(error){showMessage('import-message',error.message,'error');}
  finally{button.disabled=false;}
});

toggleFields(); refreshAll(); setInterval(refreshAll,3000);
