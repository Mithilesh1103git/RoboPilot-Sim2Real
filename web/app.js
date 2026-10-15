const byId = (id) => document.getElementById(id);
const events = byId("events");

function addEvent(message) {
  const empty = events.querySelector(".empty-event");
  if (empty) empty.remove();
  const li = document.createElement("li");
  li.textContent = `${new Date().toLocaleTimeString()} — ${message}`;
  events.prepend(li);
}

byId("runTask").addEventListener("click", async () => {
  const instruction = byId("instruction").value.trim();
  if (instruction.length < 3) {
    byId("taskMessage").textContent = "Enter a longer instruction.";
    return;
  }
  byId("runTask").disabled = true;
  try {
    const response = await fetch("/api/tasks", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({instruction, scenario: byId("scenario").value})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Task submission failed");
    byId("taskMessage").textContent = `${data.status}: ${data.message}`;
    addEvent(`Task ${data.task_id} accepted (${data.status})`);
  } catch (error) {
    byId("taskMessage").textContent = `API error: ${error.message}`;
    addEvent(`API error: ${error.message}`);
  } finally {
    byId("runTask").disabled = false;
  }
});

byId("resetSim").addEventListener("click", async () => {
  try {
    const response = await fetch("/api/simulation/reset", {method: "POST"});
    const data = await response.json();
    byId("taskMessage").textContent = data.message;
    addEvent(`Simulation reset: ${data.status}`);
  } catch (error) {
    byId("taskMessage").textContent = `Reset error: ${error.message}`;
  }
});

async function refreshTelemetry() {
  try {
    const response = await fetch("/api/telemetry");
    const data = await response.json();
    byId("connection").textContent = data.simulation.connected ? "Connected" : "Offline";
    byId("simTime").textContent = data.simulation.connected ? `${data.simulation.time.toFixed(2)} s` : "—";
    byId("currentAction").textContent = data.current_action || "—";
  } catch {
    byId("connection").textContent = "API unavailable";
  }
}
refreshTelemetry();
