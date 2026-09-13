"use strict";

let token = "", state = {objects: [], policy: {}}, catalog = {releases: [], actor: {}}, session = null, retrieval = null, editing = null, anchors = [];
const controllers = new Set(), objectURLs = new Set();
const $ = id => document.getElementById(id);
const has = role => (catalog.actor.roles || []).includes(role);
const byKind = kind => state.objects.filter(item => item.kind === kind);
const make = (tag, text = "", cls = "") => { const node = document.createElement(tag); node.textContent = text; if (cls) node.className = cls; return node; };
const expiry = value => { if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) throw Error("Choose an expiry date."); return value + "T23:59:59+00:00"; };
const lines = value => value.split("\n").map(x => x.trim()).filter(Boolean);
const labels = {pass: "✓ Passed", review: "! Needs review", fail: "× Failed"};
const errors = {
  governance_not_configured: "Governance is not configured. An operator must provision access and the independent audit key.",
  invalid_governance_credentials: "The governance access token was not recognized.",
  role_not_authorized: "Your account is not authorized for this action.",
  clinical_authority_not_provisioned: "Clinical review authority has not been provisioned for your account.",
  superseded_revision: "The recommendation has changed. Review and publish the current revision before using it.",
  source_superseded: "A newer source version exists. The older evidence cannot be used for a new answer.",
  release_revoked: "This release has been withdrawn. Start a new session with an eligible release.",
  durable_audit_unavailable: "Durable audit storage is unavailable. No governed action was completed.",
  audit_integrity_failure: "The audit integrity check failed. An operator must investigate before work continues.",
  language_not_reviewed: "No clinically reviewed wording is available in this language. No automatic language fallback was used.",
  review_policy_changed_retrieve_again: "The review policy changed. Make a fresh retrieval request.",
  request_context_review_required: "Confirm that you reviewed sufficient authorized context for this exact request.",
  session_inr_budget_exhausted: "The session INR budget is exhausted.",
  monthly_inr_budget_exhausted: "The monthly INR budget is exhausted.",
  inr_budget_not_configured: "An operator must configure INR limits before paid work.",
  payload_not_retained: "Request text and context were not retained, or have expired. Obtain authorized context before making a clinical review decision."
};

function status(text, error = false) { $("status").textContent = text; $("status").classList.toggle("error", error); }
async function attempt(action) {
  try { status("Working…"); await action(); status("Completed. Automated checks do not establish clinical correctness."); }
  catch (error) { if (error.name !== "AbortError") status(errors[error.message] || error.message.replaceAll("_", " "), true); }
}
async function api(operation, data = {}) {
  const controller = new AbortController(); controllers.add(controller);
  try {
    const response = await fetch("/api/governance/" + operation, {method: "POST", signal: controller.signal,
      headers: {Authorization: "Bearer " + token, "Content-Type": "application/json"}, body: JSON.stringify(data)});
    const result = await response.json(); if (!response.ok) throw Error(result.error || "request_failed"); return result;
  } finally { controllers.delete(controller); }
}
function setOptions(id, items, text, first = null) {
  const select = $(id), previous = select.value; select.replaceChildren();
  if (first !== null) { const option = make("option", first); option.value = ""; select.append(option); }
  for (const item of items) { const option = make("option", text(item)); option.value = item.id; select.append(option); }
  if ([...select.options].some(option => option.value === previous)) select.value = previous;
}
function showPanel(id) {
  document.querySelectorAll(".tab-panel").forEach(panel => { panel.hidden = panel.id !== id; });
  document.querySelectorAll("nav button").forEach(button => { const active = button.dataset.panel === id; button.classList.toggle("active", active); button.setAttribute("aria-pressed", String(active)); });
}
function detail(title, value) { const node = make("details"); node.append(make("summary", title), make("pre", typeof value === "string" ? value : JSON.stringify(value, null, 2))); return node; }
function action(title, callback, cls = "secondary") { const button = make("button", title, cls); button.type = "button"; button.onclick = () => attempt(callback); return button; }
function empty(container, text) { container.append(make("p", text, "empty")); }
function renderConditions(node) {
  if (node.children?.length) return (node.op === "all" ? "All of: " : node.op === "any" ? "Any of: " : "Not: ") + node.children.map(renderConditions).join("; ");
  return node.field.replaceAll("_", " ") + " " + ({eq: "equals", ne: "does not equal", gt: "is above", gte: "is at least", lt: "is below", lte: "is at most", in: "is one of", not_in: "is not one of"}[node.op] || node.op) + " " + JSON.stringify(node.value);
}

async function refresh() {
  catalog = await api("catalog");
  if (!token) return;
  if ((catalog.actor.roles || []).some(role => ["editor", "clinician", "publisher", "auditor", "operator"].includes(role))) state = await api("snapshot");
  else state = {objects: [], policy: {}};
  if (!token) return;
  $("identity").textContent = catalog.actor.id + " · " + (catalog.actor.roles || []).join(", ");
  setOptions("release-select", catalog.releases, r => r.id.slice(0, 25) + " · " + r.member_count + " recommendations", "Choose a release");
  setOptions("validation-release", catalog.releases, r => r.id, "Choose a release");
  const sources = byKind("source");
  setOptions("source-import", byKind("kl4a_import"), item => (item.manifest.title || item.manifest.id) + " · " + item.id.slice(-8), "New KL4A import");
  setOptions("draft-source", sources, s => s.title + " · " + s.version, "Choose a source");
  const families = [...new Map(sources.map(s => [s.family_id, {id: s.family_id, title: s.title}])).values()];
  setOptions("source-family", families, s => "New version of: " + s.title, "New source family");
  $("source-list").replaceChildren();
  for (const source of sources) { const row = make("div"); row.append(make("h3", source.title), make("p", "Version " + source.version, "muted"), detail("Coverage gaps and source fingerprint", {gaps: source.coverage_gaps, id: source.id, sha256: source.original_sha256})); $("source-list").append(row); }
  if (!sources.length) empty($("source-list"), "No sources yet. Start with an authored synthetic protocol or a source you are authorized to use.");
  $("review-mode").value = state.policy.mode || "off"; $("sample-percent").value = state.policy.sample_percent ?? 10; $("retain-seconds").value = state.policy.retain_request_seconds ?? 0;
  for (const formId of ["policy-form", "budget-form", "rate-form"]) $(formId).querySelectorAll("button").forEach(button => { button.disabled = !has("operator"); });
  const budget = byKind("budget_policy")[0];
  $("budget-state").textContent = budget ? "Session: ₹" + (budget.session_limit_paise / 100).toFixed(2) + "\nMonthly: ₹" + (budget.monthly_limit_paise / 100).toFixed(2) + "\nRates and actual billed amounts remain separate." : "INR limits are not configured. Paid governed operations are blocked.";
  renderReview(); renderPending(); renderImports(); renderAudit();
  $("validation-records").replaceChildren();
  for (const record of byKind("validation")) $("validation-records").append(detail(record.category + " · " + record.outcome + " · " + record.actor, record));
  if (!has("editor") && !has("clinician")) $("source-form").querySelector("button").disabled = true;
}

function renderReview() {
  const queue = $("recommendations"); queue.replaceChildren();
  const items = byKind("recommendation").sort((a, b) => Number(a.checks.find(c => c.key === "approval")?.status === "pass") - Number(b.checks.find(c => c.key === "approval")?.status === "pass"));
  $("review-count").textContent = items.filter(r => r.checks.some(c => c.key !== "applicability" && c.status !== "pass")).length + " need attention";
  if (!items.length) empty(queue, "No locally authored recommendations yet. Imported knowledge appears below as unreviewed candidates.");
  for (const item of items) {
    const card = make("article", "", "review-card"), heading = make("div", "", "card-heading");
    const approved = item.checks.find(c => c.key === "approval")?.status === "pass";
    heading.append(make("h3", item.statement), make("span", "Revision " + item.revision + " · " + (approved ? "Clinical decision recorded" : "Needs clinical review"), "card-state")); card.append(heading);
    const comparison = make("div", "", "comparison"), evidence = make("section"), interpretation = make("section");
    evidence.append(make("h4", "Source evidence"));
    for (const anchor of item.evidence) {
      const source = byKind("source").find(s => s.id === anchor.source_id);
      evidence.append(make("p", (source?.title || anchor.source_id) + " · " + (source?.version || "") + (anchor.page ? " · Page " + anchor.page : "") + " · " + anchor.role, "evidence-label"));
      const passage = make("div", "", "passage");
      if (source) { const chars = [...source.normalized_text]; passage.append(document.createTextNode(chars.slice(Math.max(0, anchor.start - 120), anchor.start).join("")), make("mark", anchor.excerpt), document.createTextNode(chars.slice(anchor.end, anchor.end + 120).join(""))); }
      else passage.textContent = anchor.excerpt;
      evidence.append(passage);
      if (source?.media_type === "application/pdf" && anchor.page) evidence.append(action("View original page", async () => {
        const response = await fetch("/api/governance/source/" + source.id + "/page/" + anchor.page, {headers: {Authorization: "Bearer " + token}});
        if (!response.ok) throw Error("Source preview is unavailable.");
        const url = URL.createObjectURL(await response.blob()); objectURLs.add(url); const img = make("img", "", "source-preview"); img.src = url; img.alt = source.title + ", page " + anchor.page; evidence.append(img);
      }));
      if (source?.coverage_gaps.length) evidence.append(detail("Source coverage gaps to review", source.coverage_gaps));
    }
    interpretation.append(make("h4", "Proposed interpretation"), make("p", item.statement), make("p", "Applies when: " + renderConditions(item.condition)));
    for (const [lang, wording] of Object.entries(item.explanations)) interpretation.append(make("h4", "Explanation · " + lang), make("p", wording));
    interpretation.append(detail("Qualifiers, exceptions and alternatives", {qualifiers: item.qualifiers, exceptions: item.exceptions, alternatives: item.alternatives}));
    comparison.append(evidence, interpretation); card.append(comparison);
    const checks = make("ul", "", "checks");
    for (const check of [...item.checks].sort((a, b) => ({fail: 0, review: 1, pass: 2}[a.status]) - ({fail: 0, review: 1, pass: 2}[b.status]))) { const row = make("li", "", "check " + check.status), block = make("details"); block.append(make("summary", labels[check.status] + " · " + check.label), make("p", check.reasons.join(" "))); row.append(block); checks.append(row); }
    card.append(checks);
    if (has("clinician") && catalog.actor.clinical_authority) {
      const form = make("form", "", "review-actions"), reason = make("textarea"), label = make("label", "Reason for this clinical decision"), attest = make("input"), attestLabel = make("label", "", "check-label");
      reason.required = true; reason.rows = 2; label.append(reason); attest.type = "checkbox";
      attestLabel.append(attest, document.createTextNode("I reviewed the exact evidence, interpretation, exceptions and coverage gaps for this revision."));
      const decision = make("select"); decision.setAttribute("aria-label", "Clinical decision");
      for (const [value, title] of [["deferred", "Request correction"], ["approved", "Approve this revision"], ["rejected", "Reject"]]) { const option = make("option", title); option.value = value; decision.append(option); }
      const button = make("button", "Record clinical decision"); form.append(label, attestLabel, decision, button);
      form.onsubmit = event => { event.preventDefault(); attempt(async () => { if (decision.value === "approved" && !attest.checked) throw Error("Confirm your clinical evidence review first."); await api("review", {id: item.id, expected_revision: item.revision, decision: decision.value, rationale: reason.value, semantic_checked: attest.checked, exceptions_checked: attest.checked, coverage_acknowledged: attest.checked}); await refresh(); }); }; card.append(form);
    } else card.append(make("p", "Clinical approval requires separately provisioned clinical authority.", "muted"));
    if (has("editor") || has("clinician")) card.append(action("Edit as a new proposed revision", async () => { editDraft(item); showPanel("sources"); $("draft-heading").scrollIntoView({behavior: "smooth"}); }));
    if (has("publisher")) { const publish = action("Publish this approved revision", async () => { const release = await api("publish", {ids: [item.id], purposes: ["information"], valid_until: item.valid_until}); await refresh(); status("Published immutable release " + release.id); }, "primary"); publish.disabled = item.checks.some(c => c.key !== "applicability" && c.status !== "pass"); card.append(publish); }
    card.append(action("Compare revision history", async () => { const history = await api("history", {id: item.id}); const panel = make("div", "", "comparison"); for (const version of history.versions.slice(-2)) { const part = make("section"); part.append(make("h4", "Revision " + version.revision + (version.revision === item.revision ? " · Current proposal" : " · Previous version")), make("p", version.statement), detail("Full revision and evidence", version)); panel.append(part); } card.append(panel); }));
    card.append(detail("Identifiers and authenticated review history", {id: item.id, content_hash: item.hash, reviews: byKind("review").filter(r => r.target_id === item.id)})); queue.append(card);
  }
}

function renderPending() {
  const container = $("pending"); container.replaceChildren();
  for (const request of byKind("retrieval").filter(r => r.review_status === "pending")) {
    const card = make("article", "", "panel"); card.append(make("h3", "Request " + request.id.slice(-10)), make("p", "Mode: " + request.review_mode + " · Caller: " + request.owner), detail("Selection and applicability decisions", request.decisions));
    card.append(action("Inspect retained request context", async () => { const result = await api("request_payload", {id: request.id}); card.append(detail("Authorized context, expires " + result.expires_at, result.payload)); }));
    const form = make("form"), reason = make("textarea"), label = make("label", "Decision rationale"), attest = make("input"), consent = make("label", "", "check-label"), decision = make("select");
    label.append(reason); reason.required = true; attest.type = "checkbox"; consent.append(attest, document.createTextNode("I reviewed sufficient authorized context for this exact request."));
    for (const value of ["rejected", "approved"]) { const option = make("option", value === "approved" ? "Approve this request" : "Reject this request"); option.value = value; decision.append(option); }
    decision.setAttribute("aria-label", "Request decision"); form.append(label, consent, decision, make("button", "Record request review"));
    form.onsubmit = event => { event.preventDefault(); attempt(async () => { await api("review_retrieval", {id: request.id, decision: decision.value, rationale: reason.value, context_reviewed: attest.checked}); await refresh(); }); }; card.append(form); container.append(card);
  }
  if (!container.children.length) empty(container, "No pending request reviews. Auditing remains mandatory with review disabled.");
}
function renderImports() {
  $("imports").replaceChildren(); for (const item of byKind("import_candidate")) { const row = make("article", "", "panel"); row.append(make("h3", item.metadata.frontmatter.title || "Imported candidate"), make("p", "Unreviewed locally · upstream label: " + (item.upstream_review_status || "unknown")), detail("Preserved candidate and coverage gaps", item)); row.append(action("Use evidence in a local draft", async () => { resetDraft(); anchors = item.anchors.map(a => ({...a, role: "passage"})); $("draft-source").value = item.source_id; renderSelection(); renderSpans(); showPanel("sources"); $("draft-heading").scrollIntoView({behavior: "smooth"}); })); $("imports").append(row); }
  if (!$("imports").children.length) empty($("imports"), "No imported KL4A candidates.");
}
function renderAudit() { $("audit").replaceChildren(); for (const event of state.audit || []) { const row = detail(event.operation + " · " + event.actor + " · " + new Date(event.at).toLocaleString(), JSON.parse(event.detail)); row.className = "audit-row"; $("audit").append(row); } }
function renderSelection() { $("source-selection").value = byKind("source").find(s => s.id === $("draft-source").value)?.normalized_text || ""; }
function renderSpans() { $("spans").replaceChildren(...anchors.map(a => make("li", a.role + ": " + a.excerpt.slice(0, 200)))); }
function resetDraft() { editing = null; anchors = []; $("draft-form").reset(); $("draft-heading").textContent = "Author a recommendation"; $("condition-field").required = $("condition-value").required = true; renderSelection(); renderSpans(); }
function editDraft(item) { resetDraft(); editing = item; anchors = item.evidence.map(({offsets, ...anchor}) => anchor); $("draft-heading").textContent = "Edit revision " + item.revision + " as a new proposal"; $("draft-source").value = anchors[0].source_id; renderSelection(); renderSpans(); $("statement").value = item.statement; $("condition-json").value = JSON.stringify(item.condition, null, 2); $("condition-field").required = $("condition-value").required = false; $("qualifiers-json").value = JSON.stringify(item.qualifiers, null, 2); $("exceptions").value = item.exceptions.join("\n"); $("alternatives").value = item.alternatives.join("\n"); $("keywords").value = item.keywords.join(", "); $("draft-expiry").value = item.valid_until.slice(0, 10); const lang = Object.keys(item.explanations)[0] || "en"; $("explanation-language").value = lang; $("explanation").value = item.explanations[lang] || ""; $("high-risk").checked = item.high_risk; }

$("login-form").onsubmit = event => { event.preventDefault(); attempt(async () => { token = $("token").value; await refresh(); $("token").value = ""; $("login").hidden = true; $("workspace").hidden = false; }); };
$("logout").onclick = () => { controllers.forEach(c => c.abort()); objectURLs.forEach(url => URL.revokeObjectURL(url)); objectURLs.clear(); token = ""; state = {objects: [], policy: {}}; catalog = {releases: [], actor: {}}; session = retrieval = null; $("workspace").hidden = true; $("login").hidden = false; $("token").value = ""; ["recommendations", "source-list", "pending", "imports", "audit", "answer", "validation-records"].forEach(id => $(id).replaceChildren()); status("Signed out."); };
$("refresh").onclick = () => attempt(refresh);
document.querySelectorAll("nav button").forEach(button => { button.onclick = () => showPanel(button.dataset.panel); });
$("draft-source").onchange = renderSelection;
$("condition-json").oninput = () => { $("condition-field").required = $("condition-value").required = !$("condition-json").value.trim(); };
$("cancel-edit").onclick = resetDraft;
$("clear-spans").onclick = () => { anchors = []; renderSpans(); };
$("add-span").onclick = () => attempt(async () => { const source = byKind("source").find(s => s.id === $("draft-source").value), area = $("source-selection"); if (!source || area.selectionStart === area.selectionEnd) throw Error("Select the exact source text before attaching evidence."); const start = [...area.value.slice(0, area.selectionStart)].length, end = [...area.value.slice(0, area.selectionEnd)].length; const page = source.pages.find(p => p.start <= start && end <= p.end); anchors.push({source_id: source.id, normalized_sha256: source.normalized_sha256, start, end, excerpt: [...area.value].slice(start, end).join(""), role: $("anchor-role").value, ...(page ? {page: page.page} : {})}); renderSpans(); });
$("source-form").onsubmit = event => { event.preventDefault(); attempt(async () => { const file = $("source-file").files[0]; if (!file || file.size > 20 * 1024 * 1024) throw Error("Choose a source no larger than 20 MB."); const metadata = {title: $("source-title").value, version: $("source-version").value, update_import_id: $("source-import").value || null, ...( $("source-family").value ? {family_id: $("source-family").value} : {}), rights: {authorized: $("source-authorized").checked, authorization_reference: $("source-reference").value, purposes: ["information"], external_processing: false, redistribution: false, valid_until: expiry($("source-expiry").value)}}; const type = file.name.toLowerCase().endsWith(".zip") ? "application/zip" : file.name.toLowerCase().endsWith(".pdf") ? "application/pdf" : file.name.toLowerCase().endsWith(".md") ? "text/markdown" : "text/plain"; const response = await fetch("/api/governance/ingest", {method: "POST", headers: {Authorization: "Bearer " + token, "Content-Type": type, "X-Source-Metadata": encodeURIComponent(JSON.stringify(metadata)), "X-Source-Metadata-Format": "uri"}, body: file}); const result = await response.json(); if (!response.ok) throw Error(result.error); await refresh(); }); };
$("draft-form").onsubmit = event => { event.preventDefault(); attempt(async () => { if (!anchors.length) throw Error("Attach at least one exact source passage."); let value = $("condition-value").value; try { value = JSON.parse(value); } catch (_) {} const condition = $("condition-json").value.trim() ? JSON.parse($("condition-json").value) : {op: "eq", field: $("condition-field").value, value}; const draft = {statement: $("statement").value, evidence: anchors, condition, qualifiers: JSON.parse($("qualifiers-json").value || "{}"), exceptions: lines($("exceptions").value), alternatives: lines($("alternatives").value), purposes: editing?.purposes || ["information"], explanations: {...(editing?.explanations || {}), [$("explanation-language").value]: $("explanation").value}, keywords: $("keywords").value.split(",").map(s => s.trim()).filter(Boolean), high_risk: $("high-risk").checked, valid_until: expiry($("draft-expiry").value), import_metadata: editing?.import_metadata || {}}; await api(editing ? "edit" : "propose", {draft, ...(editing ? {id: editing.id, expected_revision: editing.revision} : {})}); resetDraft(); await refresh(); showPanel("review"); }); };
$("policy-form").onsubmit = event => { event.preventDefault(); attempt(async () => { await api("policy", {expected_revision: state.policy.revision || 0, policy: {mode: $("review-mode").value, sample_percent: Number($("sample-percent").value), trigger_high_risk: true, retain_request_seconds: Number($("retain-seconds").value), session_ttl_seconds: state.policy.session_ttl_seconds || 3600}}); await refresh(); }); };
function paise(value) { if (!/^\d+(\.\d{1,2})?$/.test(value)) throw Error("Enter an INR amount with at most two decimal places."); const [whole, fraction = ""] = value.split("."); const amount = Number(whole) * 100 + Number(fraction.padEnd(2, "0")); if (!Number.isSafeInteger(amount)) throw Error("Amount is too large."); return amount; }
$("budget-form").onsubmit = event => { event.preventDefault(); attempt(async () => { await api("budget_configure", {expected_revision: byKind("budget_policy")[0]?.revision || 0, session_limit_paise: paise($("session-inr").value), monthly_limit_paise: paise($("monthly-inr").value)}); await refresh(); }); };
$("rate-form").onsubmit = event => { event.preventDefault(); attempt(async () => { await api("budget_rate", {service: $("rate-service").value, version: $("rate-version").value, unit: $("rate-unit").value, paise_per_unit: $("rate-price").value, billing_reference: $("rate-reference").value, valid_until: expiry($("rate-expiry").value)}); await refresh(); }); };
$("revoke-form").onsubmit = event => { event.preventDefault(); attempt(async () => { if (!confirm("Withdraw this source, recommendation or release? Its historical records remain, but dependent answers may become ineligible.")) return; await api("revoke", {id: $("revoke-id").value, reason: $("revoke-reason").value}); await refresh(); }); };
$("conflict-form").onsubmit = event => { event.preventDefault(); attempt(async () => { const id = $("conflict-id").value.trim(); await api("conflict", id ? {id, expected_revision: Number($("conflict-revision").value), status: "resolved", reason: $("conflict-reason").value} : {targets: $("conflict-targets").value.split(",").map(s => s.trim()).filter(Boolean), reason: $("conflict-reason").value}); await refresh(); }); };
$("verify-audit").onclick = () => attempt(async () => { const result = await api("audit_verify"); $("audit").prepend(make("p", "Stored chain verified. This does not certify institutional compliance or an external checkpoint.", "boundary")); });
$("session-form").onsubmit = event => { event.preventDefault(); attempt(async () => { session = await api("session", {release_id: $("release-select").value, purpose: "information"}); retrieval = null; $("session-state").textContent = "Session " + session.id + " · release pinned · epoch " + session.epoch; $("answer").textContent = ""; }); };
async function prepareAnswer() { const result = await api("answer", {id: retrieval.id}); $("answer").textContent = result.answer || errors[result.status] || result.status.replaceAll("_", " "); if (result.status === "success") await api("delivery", {id: result.id, status: "displayed"}); }
$("query-form").onsubmit = event => { event.preventDefault(); attempt(async () => { if (!session) throw Error("Start a governed session first."); retrieval = await api("retrieve", {session_id: session.id, epoch: session.epoch, request_id: crypto.randomUUID(), query: $("query").value, context: JSON.parse($("query-context").value), language: $("query-language").value, request_review: $("request-review").checked, consent_to_retain: $("retain-query").checked}); $("retrieval-details").textContent = JSON.stringify(retrieval.decisions, null, 2); $("check-answer").hidden = false; await prepareAnswer(); }); };
$("check-answer").onclick = () => attempt(prepareAnswer);
$("interrupt").onclick = () => attempt(async () => { if (!session) throw Error("No active session."); session = await api("interrupt", {id: session.id, epoch: session.epoch}); $("answer").textContent = "Pending output invalidated. Submit a fresh query."; $("session-state").textContent = "Session " + session.id + " · epoch " + session.epoch; });
$("validation-form").onsubmit = event => { event.preventDefault(); attempt(async () => { const release = catalog.releases.find(r => r.id === $("validation-release").value); if (!release) throw Error("Choose a release."); await api("validation_record", {release_id: release.id, release_hash: release.manifest_hash, category: $("validation-category").value, performed: $("validation-performed").checked, outcome: $("validation-outcome").value, evidence_reference: $("validation-reference").value, limitations: $("validation-limitations").value, configuration: JSON.parse($("validation-configuration").value)}); await refresh(); }); };
fetch("/api/governance/health").then(r => r.json()).then(result => { $("mode").textContent = result.configured ? (result.mode === "strict" ? "Strict boundary enabled" : "Parallel mode · legacy routes unchanged") : "Not configured"; }).catch(() => { $("mode").textContent = "Service unavailable"; });
