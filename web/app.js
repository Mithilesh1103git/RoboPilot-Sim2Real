const byId = (id) => document.getElementById(id);
const events = byId('events');
const terminalStates = new Set(['succeeded', 'failed', 'stopped']);
const activeTasks = new Set();
let robotProfiles = [];
let telemetryTimer = null;

function addEvent(message) {
  if (!events) return;
  const empty = events.querySelector('.empty-event');
  if (empty) empty.remove();
  const li = document.createElement('li');
  li.textContent = `${new Date().toLocaleTimeString()} — ${message}`;
  events.prepend(li);
  while (events.children.length > 50) events.lastElementChild.remove();
}

function setMessage(message) {
  const target = byId('taskMessage');
  if (target) target.textContent = message;
}

async function readResponse(response) {
  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('application/json')
    ? await response.json()
    : { detail: await response.text() };
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail || data);
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return data;
}

async function pollTask(taskId) {
  activeTasks.add(taskId);
  try {
    while (true) {
      const response = await fetch(`/tasks/${encodeURIComponent(taskId)}`, { cache: 'no-store' });
      const task = await readResponse(response);
      setMessage(`Task ${task.status}: ${task.message || task.task_id}`);
      if (terminalStates.has(task.status)) {
        addEvent(`Task ${task.task_id}: ${task.status}${task.error ? ` — ${task.error}` : ''}`);
        if (task.result) {
          const result = task.result;
          if (typeof result.duration_seconds === 'number') {
            addEvent(`Execution duration: ${result.duration_seconds.toFixed(2)} s; attempts: ${result.attempts ?? '—'}`);
          }
        }
        return task;
      }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  } catch (error) {
    setMessage(`Could not retrieve task status: ${error.message}`);
    addEvent(`Task polling error: ${error.message}`);
    return null;
  } finally {
    activeTasks.delete(taskId);
  }
}

byId('runTask')?.addEventListener('click', async () => {
  const instruction = byId('instruction')?.value.trim() || '';
  if (instruction.length < 3) {
    setMessage('Enter a longer instruction.');
    return;
  }
  const button = byId('runTask');
  button.disabled = true;
  try {
    const response = await fetch('/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ instruction, scenario: byId('scenario')?.value || 'pick_and_place', robot_name: byId('robotName')?.value || 'panda' }),
    });
    const task = await readResponse(response);
    setMessage(`Task ${task.status}: ${task.message || 'Task accepted.'}`);
    addEvent(`Task ${task.task_id} accepted (${task.status})`);
    await pollTask(task.task_id);
  } catch (error) {
    setMessage(`API error: ${error.message}`);
    addEvent(`API error: ${error.message}`);
  } finally {
    button.disabled = false;
  }
});

byId('resetSim')?.addEventListener('click', async () => {
  const button = byId('resetSim');
  button.disabled = true;
  try {
    const response = await fetch('/simulation/reset', { method: 'POST' });
    const data = await readResponse(response);
    setMessage(data.message || 'Simulation reset.');
    addEvent(`Simulation reset: ${data.status || 'ok'}`);
    await refreshTelemetry();
  } catch (error) {
    setMessage(`Reset error: ${error.message}`);
    addEvent(`Reset error: ${error.message}`);
  } finally {
    button.disabled = false;
  }
});

async function loadRobots() {
  const select = byId('robotName');
  if (!select) return;
  try {
    const response = await fetch('/robots', {cache:'no-store'});
    const data = await readResponse(response);
    robotProfiles = data.items || [];
    const previous = select.value;
    select.replaceChildren();
    for (const robot of robotProfiles) {
      const option = document.createElement('option');
      option.value = robot.id;
      option.textContent = robot.display_name || robot.id;
      select.append(option);
    }
    if (robotProfiles.some(robot => robot.id === previous)) select.value = previous;
  } catch (error) {
    addEvent(`Could not load robot profiles: ${error.message}`);
  }
}
loadRobots();

async function refreshTelemetry() {
  try {
    const response = await fetch('/api/v1/telemetry', { cache: 'no-store' });
    const data = await readResponse(response);
    const simulation = data.simulation || {};
    byId('connection').textContent = simulation.connected ? 'Connected' : 'Offline';
    byId('simTime').textContent = simulation.connected && Number.isFinite(Number(simulation.time))
      ? `${Number(simulation.time).toFixed(2)} s` : '—';
    byId('currentAction').textContent = data.current_action || '—';
    const statusPill = document.querySelector('.status-pill');
    if (statusPill) statusPill.lastChild.textContent = simulation.connected ? ' Simulation connected' : ' API ready / simulation idle';
    for (const event of (data.events || []).slice(-5)) {
      const key = `${event.task_id}:${event.status}:${event.message}`;
      if (!refreshTelemetry.seen.has(key)) {
        refreshTelemetry.seen.add(key);
        addEvent(`Task ${event.task_id}: ${event.status} — ${event.message}`);
      }
    }
  } catch (error) {
    byId('connection').textContent = 'API unavailable';
    byId('simTime').textContent = '—';
    byId('currentAction').textContent = '—';
    const statusPill = document.querySelector('.status-pill');
    if (statusPill) statusPill.lastChild.textContent = ' API unavailable';
  }
}
refreshTelemetry.seen = new Set();
refreshTelemetry();
telemetryTimer = setInterval(refreshTelemetry, 2000);
window.addEventListener('beforeunload', () => {
  if (telemetryTimer !== null) clearInterval(telemetryTimer);
});
