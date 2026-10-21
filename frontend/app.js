(() => {
  const API = (window.ROADGUARD_API_URL || "").replace(/\/$/, "");
  const $ = (selector) => document.querySelector(selector);
  const setText = (selector, value) => { const node = $(selector); if (node) node.textContent = value; };
  const esc = (value) => String(value ?? "").replace(/[&<>"']/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;", "'":"&#39;"}[char]));
  let reportsTotal = 0;
  let loadedCount = 100;
  let districtById = new Map();
  let reportLayer = null;
  const filterStatuses = ["Submitted","Pending Validation","Validated","Assigned","Repair Scheduled","Repair In Progress","Repair Completed — Pending Verification","Verified and Closed","Rejected","Reopened"];
  async function request(path, options = {}) {
    let response;
    try { response = await fetch(`${API}${path}`, options); }
    catch (_) { throw new Error("Can’t reach the RoadGuard API. Check that the backend is running and CORS_ORIGINS includes this website."); }
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `Request failed (${response.status}).`);
    return body;
  }
  function setApiStatus(online) {
    const node = $("#api-status"); if (!node) return;
    node.textContent = online ? "Connected" : "Unavailable";
    node.parentElement.querySelector("i").style.background = online ? "#58ae75" : "#d46a63";
  }
  async function loadDashboard() {
    try {
      const data = await request("/api/dashboard"); setApiStatus(true);
      setText("#stat-total", data.total_reports);
      setText("#stat-open", data.open_reports);
      setText("#stat-closed", data.verified_reports);
    } catch (error) { setApiStatus(false); console.error(error); }
  }
  function addDistrictOptions(registry) {
    districtById = new Map(registry.districts.map(item => [item.id, item]));
    const form = $("select[name=district_id]");
    form.innerHTML = '<option value="">Choose a district</option>' + registry.districts.map(d => `<option value="${esc(d.id)}">${esc(d.name)}</option>`).join("");
    $("#filter-district").innerHTML = '<option value="">All districts</option>' + registry.districts.map(d => `<option value="${esc(d.id)}">${esc(d.name)}</option>`).join("");
  }
  async function loadMapAndRegistry() {
    try {
      const data = await request("/api/districts"); addDistrictOptions(data.registry);
      const count = data.diagnostics.valid_polygon_count;
      setText("#stat-districts", count);
      const missing = data.diagnostics.missing_districts.map(item => item.name);
      setText("#map-coverage", `${count}/38 valid district boundaries · ${missing.length} missing`);
      setText("#map-gap-note", missing.length ? `No boundary polygons in the source file for: ${missing.join(", ")}. These districts remain available in the report form.` : "All registry districts have valid map boundaries.");
      $("#filter-status").innerHTML = '<option value="">All statuses</option>' + filterStatuses.map(s => `<option>${esc(s)}</option>`).join("");
      if (!window.L) throw new Error("Map library did not load. Check the browser network connection.");
      const map = L.map("district-map", {scrollWheelZoom:false}).setView([10.7905,78.7047], 6.5);
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {maxZoom:18, attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'}).addTo(map);
      const layer = L.geoJSON(data.geojson, {style:{color:"#317d82",weight:1.25,fillColor:"#8dc8b7",fillOpacity:.35}, onEachFeature:(feature, item) => item.bindTooltip(esc(feature.properties.district_name), {className:"district-tooltip"})}).addTo(map);
      if (layer.getBounds().isValid()) map.fitBounds(layer.getBounds(), {padding:[10,10]});
      reportLayer = L.layerGroup().addTo(map);
      try {
        const saved = await request("/api/reports?limit=500");
        const located = saved.reports.filter(r => Number.isFinite(Number(r.latitude)) && Number.isFinite(Number(r.longitude)) && r.latitude !== null && r.longitude !== null && Math.abs(r.latitude) <= 90 && Math.abs(r.longitude) <= 180);
        located.forEach(r => L.circleMarker([r.latitude,r.longitude], {radius:6,color:"#fff",weight:2,fillColor:"#e78d57",fillOpacity:.95}).bindPopup(`<b>${esc(r.road_name)}</b><br>${esc(r.problem_type)} · ${esc(r.district)}<br><small>Reporter submitted coordinates</small><br><a href="#track-section" data-track="${esc(r.report_id)}">Track report →</a>`).addTo(reportLayer));
        if (located.length) setText("#map-coverage", `${count}/38 district boundaries · ${located.length} saved report pin${located.length===1?"":"s"}`);
      } catch (error) { console.warn("Saved report pins unavailable:",error); }
      const loading = $("#district-map .map-loading"); if (loading) loading.remove();
    } catch (error) {
      const node = $("#district-map"); if (node) node.innerHTML = `<div class="map-loading">District map unavailable: ${esc(error.message)}</div>`;
      setText("#map-coverage", "Map data unavailable.");
      $("select[name=district_id]").innerHTML = '<option value="">District list unavailable</option>';
    }
  }
  function reportRow(item) {
    const date = item.submitted_at ? new Date(item.submitted_at).toLocaleDateString(undefined,{day:"2-digit",month:"short",year:"numeric"}) : "—";
    const severityClass = String(item.severity).toLowerCase();
    return `<tr><td><a href="#track-section" class="report-link" data-track="${esc(item.report_id)}">${esc(item.report_id.slice(0,8).toUpperCase())} ↗</a></td><td>${esc(item.road_name)}<br><small>${esc(item.district)}${item.location ? ` · ${esc(item.location)}` : ""}</small></td><td>${esc(item.problem_type)}</td><td><span class="severity severity-${esc(severityClass)}">${esc(item.severity)}</span></td><td><span class="status-tag">${esc(item.status)}</span></td><td>${esc(date)}</td></tr>`;
  }
  async function loadReports(append = false) {
    const body = $("#report-rows"); if (!append) body.innerHTML = '<tr><td colspan="6" class="table-state">Loading reports…</td></tr>';
    const params = new URLSearchParams();
    for (const [element, key] of [["#filter-district","district_id"],["#filter-problem","problem_type"],["#filter-severity","severity"],["#filter-status","status"]]) if ($(element).value) params.set(key,$(element).value);
    params.set("limit", String(loadedCount));
    try {
      const data = await request(`/api/reports?${params}`); reportsTotal=data.total;
      body.innerHTML = data.reports.length ? data.reports.map(reportRow).join("") : '<tr><td colspan="6" class="table-state">No reports match these filters yet.</td></tr>';
      setText("#report-count", `${data.reports.length} of ${data.total} reports`);
      $("#load-more").hidden = data.reports.length >= data.total || data.reports.length < loadedCount;
    } catch (error) { body.innerHTML = `<tr><td colspan="6" class="table-state">${esc(error.message)}</td></tr>`; setText("#report-count", "Reports unavailable"); }
  }
  async function trackReport(id) {
    const result = $("#track-result"); result.innerHTML = "<p>Looking up report…</p>";
    try {
      const data = await request(`/api/reports/${encodeURIComponent(id)}`);
      const r = data.report;
      result.innerHTML = `<article class="track-card"><div class="track-card-head"><div><h3>${esc(r.road_name)} · ${esc(r.district)}</h3><p>${esc(r.problem_type)} · ${esc(r.severity)} · ID ${esc(r.report_id)}</p></div><span class="status-tag">${esc(r.status)}</span></div><ol class="timeline">${data.status_history.map(h=>`<li><b>${esc(h.new_status)}</b><small>${esc(new Date(h.changed_at).toLocaleString())} · ${esc(h.reason)}</small></li>`).join("")}</ol></article>`;
    } catch (error) { result.innerHTML = `<p class="track-error">${esc(error.message)}</p>`; }
  }
  $("#report-form").addEventListener("submit", async event => {
    event.preventDefault(); const form=event.currentTarget, button=$("#submit-button"), message=$("#form-message");
    const lat=form.elements.latitude.value, lon=form.elements.longitude.value;
    if ((lat && !lon) || (!lat && lon)) { message.className="form-message error"; message.textContent="Enter both coordinates, or leave both blank."; return; }
    button.disabled=true; button.textContent="Submitting…"; message.className="form-message"; message.textContent="Saving your report and analyzing the photo if the model is available…";
    try {
      const data=await request("/api/reports",{method:"POST",body:new FormData(form)});
      const ai=data.ai_analysis;
      let analysis = ai.status === "detections" ? `${ai.detections.length} model detection${ai.detections.length===1?"":"s"}.` : ai.status === "no_detections" ? "The model returned no detections." : ai.status === "low_confidence" ? "The model found only low confidence detections." : ai.message || (ai.status === "not_requested" ? "No photo was submitted; no AI analysis ran." : `AI analysis status: ${ai.status}.`);
      message.className="form-message success"; message.textContent=`Report saved. ID: ${data.report.report_id}. Status: ${data.report.status}. ${analysis}`;
      $("#track-id").value=data.report.report_id; form.reset(); $("#photo-preview").hidden=true; setText("#upload-label","Choose an image");
      await Promise.all([loadDashboard(),loadReports()]);
    } catch(error) { message.className="form-message error"; message.textContent=`Report was not submitted: ${error.message}`; }
    finally { button.disabled=false; button.innerHTML='Submit report <span>→</span>'; }
  });
  $("#photo").addEventListener("change",event=>{
    const file=event.target.files[0], preview=$("#photo-preview");
    if (!file) { preview.hidden=true; setText("#upload-label","Choose an image"); return; }
    setText("#upload-label",file.name); setText("#upload-hint",`${(file.size/1024/1024).toFixed(2)} MB`);
    if (!file.type.startsWith("image/")) return;
    const image=document.createElement("img"); image.alt="Selected report photo preview"; image.src=URL.createObjectURL(file); preview.replaceChildren(image); preview.hidden=false;
  });
  $("#track-form").addEventListener("submit",event=>{event.preventDefault();trackReport($("#track-id").value.trim());});
  document.addEventListener("click",event=>{const link=event.target.closest("[data-track]");if(link){$("#track-id").value=link.dataset.track;trackReport(link.dataset.track);}});
  ["#filter-district","#filter-problem","#filter-severity","#filter-status"].forEach(selector=>$(selector).addEventListener("change",()=>{loadedCount=100;loadReports();}));
  $("#refresh-history").addEventListener("click",event=>{event.preventDefault();loadReports();});
  $("#load-more").addEventListener("click",()=>{loadedCount=Math.min(loadedCount+100,500);loadReports();});
  $("#menu-toggle").addEventListener("click",()=>$("#sidebar").classList.toggle("open"));
  document.querySelectorAll(".nav-link").forEach(link=>link.addEventListener("click",()=>$("#sidebar").classList.remove("open")));
  loadDashboard(); loadMapAndRegistry(); loadReports();
})();
