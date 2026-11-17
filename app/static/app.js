/* Streaming CTC workbench UI.
 *
 * Every server response carries a monotonically increasing revision.  The UI
 * remembers the revision it currently shows and ignores any response that is
 * older, so a slow advance or undo replay can never overwrite a newer
 * confirm / undo.  Mutating requests are additionally serialised in a small
 * client-side queue, and while a request is in flight the saved subtitle and
 * progress stay on screen until the answer arrives (or an error is shown).
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const els = {
    connStatus: $("conn-status"),
    setupPanel: $("setup-panel"),
    vocabFile: $("vocab-file"),
    framesFile: $("frames-file"),
    beamWidth: $("beam-width"),
    beamOut: $("beam-out"),
    chunkSize: $("chunk-size"),
    chunkOut: $("chunk-out"),
    loadBtn: $("load-btn"),
    resetBtn: $("reset-btn"),
    setupHint: $("setup-hint"),
    setupError: $("setup-error"),
    sampleButtons: $("sample-buttons"),
    workspace: $("workspace"),
    progressBar: $("progress-bar"),
    progressText: $("progress-text"),
    confirmedText: $("confirmed-text"),
    pendingText: $("pending-text"),
    confirmBest: $("confirm-best"),
    undoBtn: $("undo-btn"),
    stepBtn: $("step-btn"),
    runBtn: $("run-btn"),
    restartBtn: $("restart-btn"),
    stopBtn: $("stop-btn"),
    prefixInput: $("prefix-input"),
    applyPrefix: $("apply-prefix"),
    psText: $("ps-text"),
    psBlank: $("ps-blank"),
    psNonblank: $("ps-nonblank"),
    psTotal: $("ps-total"),
    opError: $("op-error"),
    opNotice: $("op-notice"),
    beamBody: $("beam-body"),
    beamEmpty: $("beam-empty"),
    peekFrame: $("peek-frame"),
    peekFrameOut: $("peek-frame-out"),
    peekNote: $("peek-note"),
    frameProbs: $("frame-probs"),
    historyList: $("history-list"),
    historyEmpty: $("history-empty"),
    busy: $("busy"),
    busyText: $("busy-text"),
    riskStart: $("risk-start"),
    riskStartOut: $("risk-start-out"),
    riskCount: $("risk-count"),
    riskCountOut: $("risk-count-out"),
    riskLimit: $("risk-limit"),
    riskLimitOut: $("risk-limit-out"),
    riskRun: $("risk-run"),
    riskStale: $("risk-stale"),
    riskRangeNote: $("risk-range-note"),
    riskZeroNote: $("risk-zero-note"),
    riskError: $("risk-error"),
    riskResult: $("risk-result"),
    riskEmpty: $("risk-empty"),
    riskBody: $("risk-body"),
    riskOver: $("risk-over"),
    riskMode: $("risk-mode"),
    riskMass: $("risk-mass"),
    riskMetaInterval: $("risk-meta-interval"),
    riskMetaLimit: $("risk-meta-limit"),
    riskFootnote: $("risk-footnote"),
  };

  const state = {
    sessionId: null,
    generation: 0,      // bumped whenever a new session replaces the old one
    revision: -1,       // revision of the state currently displayed
    snapshot: null,     // last state object accepted from the server
    busy: false,
    running: false,
    runAbort: false,
    pending: 0,
    samples: [],
    // Behind the scenes: the last shown state is kept while a request is in
    // flight, so a failed operation leaves the subtitles and progress intact.
    peekLive: false,
    // Length-risk analysis.  `token` identifies the newest request so a slow
    // earlier calculation can never overwrite a newer one; `stale` marks the
    // displayed result as outdated after the interval or limit changes.
    risk: {
      token: 0,
      inFlight: false,
      result: null,
      stale: false,
      zeroNote: "",
    },
  };

  // ---------------------------------------------------------------- helpers

  const fmt = (value) => {
    if (value === null || value === undefined) return "—";
    if (typeof value !== "number" || Number.isNaN(value)) return "—";
    if (!Number.isFinite(value)) return "-∞";
    return value.toFixed(4);
  };

  const pct = (value) => {
    if (!Number.isFinite(value)) return "0";
    if (value === 0) return "0";
    if (value >= 0.01) return (value * 100).toFixed(1) + "%";
    return value.toExponential(2);
  };

  function setConnection(kind, label) {
    els.connStatus.dataset.state = kind;
    els.connStatus.textContent = label;
  }

  function setBusy(on, text = "计算中…") {
    state.busy = on;
    els.busy.hidden = !on;
    els.busyText.textContent = text;
    if (on) setConnection("busy", "计算中");
    else updateStatusFromSnapshot();
    updateButtons();
  }

  // Mark an operation as in flight.  Operations are *not* blocked while
  // another one runs: the server applies them in submission order and each
  // response is revision-guarded, so advancing / undoing during a replay is
  // safe and the late replay result cannot overwrite the newer action.
  function beginOp(label) {
    state.pending += 1;
    setBusy(true, label);
  }

  function endOp() {
    state.pending = Math.max(0, state.pending - 1);
    if (state.pending === 0) setBusy(false);
  }

  function updateStatusFromSnapshot() {
    if (!state.snapshot) { setConnection("idle", "未加载"); return; }
    if (state.running) setConnection("busy", "连续推进中");
    else setConnection("ready", "就绪");
  }

  function showError(message) {
    hideNotice();
    if (!message) { els.opError.hidden = true; els.opError.textContent = ""; return; }
    els.opError.hidden = false;
    els.opError.textContent = message;
  }

  function showNotice(message) {
    if (!message) { els.opNotice.hidden = true; els.opNotice.textContent = ""; return; }
    els.opNotice.hidden = false;
    els.opNotice.textContent = message;
  }

  function hideNotice() { els.opNotice.hidden = true; els.opNotice.textContent = ""; }

  function updateButtons() {
    const ready = !!state.sessionId && !!state.snapshot;
    const atEnd = ready && state.snapshot.next_frame >= state.snapshot.frame_count;
    // Recompute (undo replay / long advance) deliberately does NOT disable
    // advance or undo: those actions are queued by the server behind the
    // running operation and applied to its result, in order.
    els.confirmBest.disabled = !ready;
    els.undoBtn.disabled = !ready || (state.snapshot.confirmed_count || 0) === 0;
    els.stepBtn.disabled = !ready || atEnd || state.running;
    els.runBtn.disabled = !ready || atEnd || state.running;
    els.restartBtn.disabled = !ready || state.running;
    els.applyPrefix.disabled = !ready;
    els.stopBtn.hidden = !state.running;
    els.resetBtn.disabled = !ready;
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    let payload = null;
    try { payload = await response.json(); } catch { payload = null; }
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : `HTTP ${response.status}`;
      const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      err.status = response.status;
      throw err;
    }
    return payload;
  }

  // Apply a server state snapshot, unless it belongs to a replaced session or
  // is older than the state already displayed.  Together with the server's
  // in-order worker this is what makes late results harmless.
  function applyState(payload, { force = false } = {}) {
    if (!payload || typeof payload.revision !== "number") return false;
    if (payload.session_id !== state.sessionId) return false;
    if (!force && payload.revision < state.revision) return false;
    state.revision = payload.revision;
    state.snapshot = payload;
    render();
    return true;
  }

  function startSession(payload, sampleName = null) {
    // Invalidate anything still in flight for the previous session.
    state.generation += 1;
    state.runAbort = true;
    state.running = false;
    state.pending = 0;
    els.stopBtn.hidden = true;
    state.sessionId = payload.session_id;
    state.revision = -1;
    state.snapshot = null;
    state.loadedSample = sampleName;
    state.peekLive = true;
    resetRiskPanel();
    applyState(payload, { force: true });
    if (state.snapshot) {
      els.peekFrame.value = "0";
      els.peekFrame.max = String(state.snapshot.frame_count);
      els.peekFrameOut.textContent = `0 / ${state.snapshot.frame_count}`;
      els.peekNote.textContent = "显示当前帧";
    }
    updateButtons();
    updateStatusFromSnapshot();
  }

  // ---------------------------------------------------------------- rendering

  function render() {
    const snap = state.snapshot;
    if (!snap) { els.workspace.hidden = true; return; }
    els.workspace.hidden = false;

    const total = snap.frame_count || 0;
    const done = snap.next_frame || 0;
    els.progressBar.style.width = total ? `${(100 * done) / total}%` : "0%";
    els.progressText.textContent = `${done} / ${total} 帧`;

    const best = snap.best_text || "";
    const prefix = snap.prefix || "";
    const pending = best.startsWith(prefix) ? best.slice(prefix.length) : best;
    els.confirmedText.textContent = prefix;
    els.pendingText.textContent = pending;

    const scores = snap.prefix_scores || {};
    els.psText.textContent = scores.text ? `“${scores.text}” → ${scores.last_char}` : "—";
    els.psBlank.textContent = fmt(scores.logp_blank);
    els.psNonblank.textContent = fmt(scores.logp_nonblank);
    els.psBlank.title = "确认时的末字 blank 路径概率（保留值，不随新帧归一化）";
    els.psNonblank.title = "确认时的末字非 blank 路径概率（保留值，不随新帧归一化）";
    els.psTotal.textContent = fmt(scores.logp);
    if (scores.text) {
      els.psTotal.title =
        `确认于第 ${scores.confirmed_at} 帧；此前缀“${scores.text}”的完整前向分数 ` +
        `为 ${fmt(scores.logp)}，其中末字前的前缀分数为 ${fmt(scores.previous_logp)}。` +
        "值保留自确认时刻，不随后续帧或再次确认归一化。";
    } else {
      els.psTotal.title = "尚未确认前缀";
    }

    renderBeams(snap, prefix);

    els.peekFrame.max = String(total);
    if (Number(els.peekFrame.value) > total) els.peekFrame.value = String(total);
    els.peekFrameOut.textContent = `${els.peekFrame.value} / ${total}`;
    els.peekNote.textContent =
      Number(els.peekFrame.value) === done ? "显示当前帧" : "历史回放（只读）";

    renderHistory(snap);
    renderFrameProbs(snap.frame_info);
    els.prefixInput.placeholder = prefix ? `当前前缀：${prefix}` : "例如：啊";
    updateRiskControls(snap);
    updateButtons();
    updateStatusFromSnapshot();
  }

  function renderBeams(snap, prefix) {
    const beams = snap.beams || [];
    if (typeof snap.revision === "number") state.peekLive = true;
    els.beamBody.innerHTML = "";
    els.beamEmpty.hidden = beams.length > 0;
    beams.forEach((beam, index) => {
      const tr = document.createElement("tr");
      if (index === 0) tr.className = "selected";

      const idx = document.createElement("td");
      idx.textContent = String(index + 1);

      const textTd = document.createElement("td");
      const span = document.createElement("span");
      span.className = "beam-text";
      const rest = beam.text.startsWith(prefix) ? beam.text.slice(prefix.length) : beam.text;
      if (prefix) {
        const pfx = document.createElement("span");
        pfx.className = "pfx";
        pfx.textContent = prefix;
        span.appendChild(pfx);
      }
      const tail = document.createElement("span");
      if (rest) tail.textContent = rest;
      else { tail.className = "empty-tag"; tail.textContent = "∅"; }
      span.appendChild(tail);
      textTd.appendChild(span);
      if (beam.text.startsWith(prefix)) textTd.title = beam.text;

      const logpTd = document.createElement("td");
      logpTd.className = "num";
      logpTd.textContent = fmt(beam.logp);

      const probTd = document.createElement("td");
      probTd.className = "num";
      probTd.textContent = pct(beam.prob);

      const blankTd = document.createElement("td");
      blankTd.className = "num";
      blankTd.textContent = fmt(beam.logp_blank);

      const nbTd = document.createElement("td");
      nbTd.className = "num";
      nbTd.textContent = fmt(beam.logp_nonblank);

      const actTd = document.createElement("td");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = "确认此候选";
      btn.title = `确认前缀：${beam.text}`;
      btn.disabled = !state.peekLive;
      btn.addEventListener("click", () => confirmPrefix(beam.text));
      actTd.appendChild(btn);

      tr.append(idx, textTd, logpTd, probTd, blankTd, nbTd, actTd);
      els.beamBody.appendChild(tr);
    });
  }

  function renderHistory(snap) {
    const list = snap.confirmations || [];
    els.historyList.innerHTML = "";
    els.historyEmpty.hidden = list.length > 0;
    list.forEach((entry, index) => {
      const li = document.createElement("li");
      const left = document.createElement("span");
      const badge = document.createElement("span");
      badge.className = "h-meta";
      badge.textContent = `#${entry.seq} @${entry.frame} 帧`;
      const prefixSpan = document.createElement("span");
      prefixSpan.className = "h-prefix";
      prefixSpan.textContent = entry.prefix_after || "(空)";
      left.append(badge, document.createTextNode("  "), prefixSpan);

      const right = document.createElement("span");
      right.className = "h-meta";
      right.textContent =
        `logP ${fmt(entry.confirmed_logp)} · blank ${fmt(entry.confirmed_logp_blank)}` +
        ` · 非 blank ${fmt(entry.confirmed_logp_nonblank)}`;

      li.append(left, right);
      if (index === list.length - 1) li.style.borderColor = "rgba(79, 209, 197, .45)";
      els.historyList.appendChild(li);
    });
  }

  function renderFrameProbs(info) {
    els.frameProbs.innerHTML = "";
    if (!info || !info.probs) return;
    const vocab = info.vocab || [];
    info.probs.forEach((value, index) => {
      const cell = document.createElement("div");
      cell.className = "fp" + (value === 0 ? " zero" : "");
      const tok = document.createElement("span");
      tok.className = "tok";
      tok.textContent = index === 0 ? "<blank>" : vocab[index];
      const val = document.createElement("span");
      val.className = "val";
      val.textContent = "  " + value.toFixed(3);
      cell.append(tok, val);
      cell.title = `log P = ${info.log_probs && info.log_probs[index] !== null ? Number(info.log_probs[index]).toFixed(4) : "-∞"}`;
      els.frameProbs.appendChild(cell);
    });
  }

  // ------------------------------------------------------------- operations

  async function withBusy(label, fn) {
    beginOp(label);
    try {
      return await fn();
    } finally {
      endOp();
    }
  }

  async function createSession(formData) {
    els.setupError.hidden = true;
    setBusy(true, "解析并创建会话…");
    try {
      const payload = await api("/api/sessions/upload", { method: "POST", body: formData, headers: {} });
      startSession(payload);
      showError("");
      hideNotice();
      els.loadBtn.disabled = false;
      setConnection("ready", "就绪");
    } catch (error) {
      els.setupError.hidden = false;
      els.setupError.textContent = error.message;
      setConnection("error", "加载失败");
    } finally {
      setBusy(false);
      els.loadBtn.disabled = false;
    }
  }

  async function advanceTo(target, { silentBusy = false } = {}) {
    if (!state.sessionId) return false;
    const sessionId = state.sessionId;
    const generation = state.generation;
    const run = async () => {
      try {
        const payload = await api("/api/advance", {
          method: "POST",
          body: JSON.stringify({
            session_id: sessionId,
            to_frame: target,
          }),
        });
        if (generation !== state.generation || sessionId !== state.sessionId) return false;
        const applied = applyState(payload);
        if (applied) { showError(""); hideNotice(); }
        return applied;
      } catch (error) {
        if (generation !== state.generation || sessionId !== state.sessionId) return false;
        showError(`推进失败：${error.message}（字幕与进度保持不变）`);
        return false;
      }
    };
    return silentBusy ? run() : withBusy("推进帧…", run);
  }

  async function confirmPrefix(prefix) {
    if (!state.sessionId) return;
    if (!prefix || prefix === (state.snapshot && state.snapshot.prefix)) {
      showError("请输入比当前已确认前缀更长的文本");
      return;
    }
    if (!state.peekLive) {
      showError("正在查看历史帧；请先把查看帧拖回当前帧再确认");
      return;
    }
    const sessionId = state.sessionId;
    const generation = state.generation;
    await withBusy("确认前缀…", async () => {
      try {
        const payload = await api("/api/confirm", {
          method: "POST",
          body: JSON.stringify({
            session_id: sessionId,
            prefix,
          }),
        });
        if (generation !== state.generation) return;
        if (applyState(payload)) {
          showError("");
          showNotice(`已确认前缀「${prefix}」。后续只保留兼容候选。`);
          els.prefixInput.value = "";
        }
      } catch (error) {
        showError(`确认失败：${error.message}（前缀与候选保持不变）`);
      }
    });
  }

  async function undo() {
    if (!state.sessionId) return;
    const sessionId = state.sessionId;
    const generation = state.generation;
    await withBusy("撤销确认并重算…", async () => {
      try {
        const payload = await api("/api/undo", {
          method: "POST",
          body: JSON.stringify({ session_id: sessionId }),
        });
        if (generation !== state.generation) return;
        if (applyState(payload)) {
          showError("");
          showNotice("已撤销上次确认，束与分数恢复到未作该次确认时应有的状态。");
          els.prefixInput.value = "";
        }
      } catch (error) {
        showError(`撤销失败：${error.message}（字幕与进度保持不变）`);
      }
    });
  }

  async function peek(frame) {
    if (!state.sessionId) return;
    try {
      const payload = await api("/api/peek", {
        method: "POST",
        body: JSON.stringify({ session_id: state.sessionId, frame }),
      });
      // Peek responses are informational only; they must not clobber the live
      // beam table while newer state is displayed.
      if (payload.revision !== state.revision) return;
      if (!state.snapshot) return;
      if (frame === state.snapshot.next_frame) {
        // Back on the live boundary: show the live state, not a replay.
        state.peekLive = true;
        renderBeams(state.snapshot, state.snapshot.prefix || "");
        renderFrameProbs(state.snapshot.frame_info);
        els.peekNote.textContent = "显示当前帧";
        return;
      }
      const prefix = payload.prefix || "";
      renderBeams({ beams: payload.beams, prefix }, prefix);
      state.peekLive = false;
      renderFrameProbs(payload.frame_info);
      els.peekNote.textContent = `历史回放：第 ${frame} 帧（只读，确认按钮无效）`;
    } catch (error) {
      if (frame === (state.snapshot ? state.snapshot.next_frame : -1)) {
        showError(`查看帧失败：${error.message}`);
      }
    }
  }

  async function runContinuously() {
    if (!state.sessionId || !state.snapshot || state.running) return;
    state.running = true;
    state.runAbort = false;
    state.busy = false;
    updateButtons();
    setConnection("busy", "连续推进中");
    while (state.running && !state.runAbort) {
      const snap = state.snapshot;
      if (!snap || snap.next_frame >= snap.frame_count) break;
      const chunk = Math.max(1, Number(els.chunkSize.value) || 40);
      const target = Math.min(snap.frame_count, snap.next_frame + chunk);
      // Advance without the modal busy overlay; the status pill shows progress.
      const ok = await advanceTo(target, { silentBusy: true });
      if (!ok) break;
      els.peekFrame.value = String(state.snapshot.next_frame);
      els.peekFrameOut.textContent =
        `${state.snapshot.next_frame} / ${state.snapshot.frame_count}`;
      els.peekNote.textContent = "显示当前帧";
      await new Promise((resolve) => setTimeout(resolve, 60));
    }
    state.running = false;
    updateButtons();
    updateStatusFromSnapshot();
  }

  // --------------------------------------------------------- length risk

  function resetRiskPanel() {
    state.risk.token += 1;
    state.risk.inFlight = false;
    state.risk.result = null;
    state.risk.stale = false;
    state.risk.zeroNote = "";
    els.riskStart.value = "0";
    els.riskCount.value = "4";
    els.riskLimit.value = "4";
    els.riskStartOut.textContent = "0";
    els.riskCountOut.textContent = "4";
    els.riskLimitOut.textContent = "4";
    els.riskStale.hidden = true;
    els.riskStale.classList.remove("over");
    els.riskRun.textContent = "计算字数风险";
    els.riskResult.hidden = true;
    els.riskEmpty.hidden = false;
    els.riskError.hidden = true;
    els.riskError.textContent = "";
    els.riskZeroNote.hidden = true;
    els.riskZeroNote.textContent = "";
    els.riskBody.innerHTML = "";
  }

  // Keep the sliders inside the frames that have actually been advanced.
  // Advancing more frames only widens the available range, so an existing
  // selection stays valid and the previous result is not invalidated.
  function updateRiskControls(snap) {
    const advanced = snap.next_frame || 0;
    const total = snap.frame_count || 0;
    const maxStart = Math.max(0, advanced - 1);

    els.riskStart.max = String(maxStart);
    if (Number(els.riskStart.value) > maxStart) els.riskStart.value = String(maxStart);
    els.riskStartOut.textContent = els.riskStart.value;

    const start = Number(els.riskStart.value);
    const maxCount = Math.min(64, Math.max(1, advanced - start));
    els.riskCount.max = String(maxCount);
    if (Number(els.riskCount.value) > maxCount) els.riskCount.value = String(maxCount);
    if (Number(els.riskCount.value) < 1) els.riskCount.value = "1";
    els.riskCountOut.textContent = els.riskCount.value;
    els.riskLimitOut.textContent = els.riskLimit.value;

    if (advanced === 0) {
      els.riskRangeNote.textContent = "请先推进至少 1 帧，再做字数风险分析。";
      els.riskRun.disabled = true;
    } else {
      const count = Number(els.riskCount.value);
      els.riskRangeNote.textContent =
        `可选区间限定在已推进的 0～${advanced} 帧边界内（音频共 ${total} 帧）；` +
        `当前分析第 ${start} 帧起连续 ${count} 帧。`;
      // Deliberately not disabled while a calculation is in flight: a newer
      // click is allowed and its token wins, so a slow earlier answer can
      // never overwrite the newer result.
      els.riskRun.disabled = false;
    }
    renderRiskStaleBadge();
  }

  function renderRiskStaleBadge() {
    const has = !!state.risk.result;
    if (!has || !state.risk.stale) {
      els.riskStale.hidden = true;
      return;
    }
    els.riskStale.hidden = false;
    els.riskStale.textContent = "结果已过期（区间或上限已改动）";
    els.riskStale.classList.remove("over");
  }

  // Any parameter change invalidates the displayed result immediately, but the
  // old table stays on screen marked as stale until a fresh answer arrives.
  function markRiskStale() {
    if (state.risk.result) {
      state.risk.stale = true;
      renderRiskStaleBadge();
    }
    els.riskError.hidden = true;
  }

  function riskParams() {
    return {
      start_frame: Number(els.riskStart.value),
      frame_count: Number(els.riskCount.value),
      max_chars: Number(els.riskLimit.value),
    };
  }

  async function runLengthRisk() {
    if (!state.sessionId || !state.snapshot) return;
    if ((state.snapshot.next_frame || 0) === 0) {
      els.riskError.hidden = false;
      els.riskError.textContent = "请先推进至少 1 帧，再做字数风险分析。";
      return;
    }
    const sessionId = state.sessionId;
    const generation = state.generation;
    const token = ++state.risk.token;
    const params = riskParams();
    state.risk.inFlight = true;
    els.riskRun.textContent = "计算中…";
    try {
      const payload = await api("/api/length-risk", {
        method: "POST",
        body: JSON.stringify({ session_id: sessionId, ...params }),
      });
      // A newer request, or a replaced session, wins: discard this reply.
      if (token !== state.risk.token || generation !== state.generation ||
          sessionId !== state.sessionId) return;
      state.risk.inFlight = false;
      els.riskRun.textContent = "计算字数风险";

      if (payload.computable === false) {
        // Keep the previous result on screen; only surface the explanation.
        state.risk.zeroNote = payload.message || "区间含全零帧，无法计算。";
        els.riskZeroNote.hidden = false;
        els.riskZeroNote.textContent = state.risk.zeroNote;
        return;
      }
      state.risk.zeroNote = "";
      els.riskZeroNote.hidden = true;
      state.risk.result = payload;
      state.risk.stale = false;
      renderRiskResult(payload);
    } catch (error) {
      if (token !== state.risk.token || generation !== state.generation ||
          sessionId !== state.sessionId) return;
      state.risk.inFlight = false;
      els.riskRun.textContent = "计算字数风险";
      els.riskError.hidden = false;
      els.riskError.textContent = `风险分析失败：${error.message}（上次结果保持不变）`;
    }
  }

  function renderRiskResult(payload) {
    els.riskResult.hidden = false;
    els.riskEmpty.hidden = true;
    els.riskStale.hidden = true;
    els.riskStale.classList.remove("over");
    els.riskError.hidden = true;

    els.riskMetaInterval.textContent =
      `${payload.start_frame}～${payload.end_frame}（${payload.frame_count} 帧）`;
    els.riskMetaLimit.textContent = String(payload.max_chars);

    const over = Number(payload.over_limit_prob) || 0;
    els.riskOver.textContent = pct(over);
    els.riskOver.classList.toggle("danger", over >= 0.3);
    els.riskOver.classList.toggle("warn", over > 0 && over < 0.3);
    els.riskOver.title =
      `折叠后字数超过 ${payload.max_chars} 字的精确路径概率（完整路径，未剪枝）。`;

    const rows = payload.lengths || [];
    const best = rows.reduce(
      (acc, row) => (row.prob > (acc ? acc.prob : -1) ? row : acc), null);
    els.riskMode.textContent = best ? `${best.length} 字（${pct(best.prob)}）` : "—";
    els.riskMass.textContent = Number.isFinite(payload.total_mass)
      ? payload.total_mass.toFixed(6) : "—";
    els.riskMass.title =
      "区间全部路径的概率质量之和；逐帧归一化输入下应等于 1，字数分布按它归一化。";

    const maxProb = Math.max(
      ...rows.map((row) => row.prob), over, 1e-12);
    els.riskBody.innerHTML = "";
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      if (best && row.length === best.length && row.prob > 0) tr.className = "is-mode";
      const lenTd = document.createElement("td");
      lenTd.textContent = `${row.length} 字`;
      const probTd = document.createElement("td");
      probTd.className = "num";
      probTd.textContent = pct(row.prob);
      const logTd = document.createElement("td");
      logTd.className = "num";
      logTd.textContent = row.log_prob === null ? "-∞" : Number(row.log_prob).toFixed(4);
      const barTd = document.createElement("td");
      barTd.className = "risk-bar-cell";
      const track = document.createElement("div");
      track.className = "risk-bar-track";
      const fill = document.createElement("div");
      fill.className = "risk-bar-fill";
      fill.style.width = `${(100 * row.prob) / maxProb}%`;
      track.appendChild(fill);
      barTd.appendChild(track);
      tr.append(lenTd, probTd, logTd, barTd);
      els.riskBody.appendChild(tr);
    });
    // Over-limit aggregate row.
    const tr = document.createElement("tr");
    tr.className = "over-limit";
    const lenTd = document.createElement("td");
    lenTd.textContent = `> ${payload.max_chars} 字（超限合计）`;
    const probTd = document.createElement("td");
    probTd.className = "num";
    probTd.textContent = pct(over);
    const logTd = document.createElement("td");
    logTd.className = "num";
    logTd.textContent = payload.over_limit_log_prob === null
      ? "-∞" : Number(payload.over_limit_log_prob).toFixed(4);
    const barTd = document.createElement("td");
    barTd.className = "risk-bar-cell";
    const track = document.createElement("div");
    track.className = "risk-bar-track";
    const fill = document.createElement("div");
    fill.className = "risk-bar-fill over-fill";
    fill.style.width = `${(100 * over) / maxProb}%`;
    track.appendChild(fill);
    barTd.appendChild(track);
    tr.append(lenTd, probTd, logTd, barTd);
    els.riskBody.appendChild(tr);

    els.riskFootnote.textContent =
      "概率由完整 CTC 路径前向 DP 精确计算（不使用剪枝后的候选）；" +
      "各行概率与超限合计之和为 1。连续同字未经过 blank 只计一个字。";
  }

  // ------------------------------------------------------------------- wiring

  els.beamWidth.addEventListener("input", () => {
    els.beamOut.textContent = els.beamWidth.value;
  });
  els.chunkSize.addEventListener("input", () => {
    els.chunkOut.textContent = els.chunkSize.value;
  });

  els.riskStart.addEventListener("input", () => {
    els.riskStartOut.textContent = els.riskStart.value;
    // Changing the start may shrink the selectable interval length.
    if (state.snapshot) updateRiskControls(state.snapshot);
    markRiskStale();
  });
  els.riskCount.addEventListener("input", () => {
    els.riskCountOut.textContent = els.riskCount.value;
    markRiskStale();
  });
  els.riskLimit.addEventListener("input", () => {
    els.riskLimitOut.textContent = els.riskLimit.value;
    markRiskStale();
  });
  els.riskRun.addEventListener("click", runLengthRisk);

  els.loadBtn.addEventListener("click", () => {
    const vocabFile = els.vocabFile.files[0];
    const framesFile = els.framesFile.files[0];
    if (!vocabFile || !framesFile) {
      els.setupError.hidden = false;
      els.setupError.textContent = "请同时选择词表文件和逐帧概率文件";
      return;
    }
    const form = new FormData();
    form.append("vocab_file", vocabFile);
    form.append("frames_file", framesFile);
    form.append("beam_width", els.beamWidth.value);
    createSession(form);
  });

  els.resetBtn.addEventListener("click", async () => {
    if (state.sessionId) {
      try {
        await api(`/api/sessions/${state.sessionId}`, { method: "DELETE" });
      } catch { /* ignore */ }
    }
    state.sessionId = null;
    state.revision = -1;
    state.snapshot = null;
    els.workspace.hidden = true;
    state.loadedSample = null;
    resetRiskPanel();
    setConnection("idle", "未加载");
    updateButtons();
  });

  els.confirmBest.addEventListener("click", () => {
    const snap = state.snapshot;
    if (!snap) return;
    const best = snap.best_text || "";
    let target = best;
    if (!best || best === snap.prefix) {
      showError("当前最佳候选与已确认前缀相同，无法继续确认");
      return;
    }
    // confirm the full best hypothesis; users can also type a shorter prefix
    confirmPrefix(target);
  });

  els.applyPrefix.addEventListener("click", () => {
    confirmPrefix(els.prefixInput.value.trim());
  });

  els.prefixInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      confirmPrefix(els.prefixInput.value.trim());
    }
  });

  els.undoBtn.addEventListener("click", undo);

  els.stepBtn.addEventListener("click", async () => {
    const snap = state.snapshot;
    if (!snap) return;
    const chunk = Math.max(1, Number(els.chunkSize.value) || 40);
    const target = Math.min(snap.frame_count, snap.next_frame + chunk);
    const ok = await advanceTo(target);
    if (ok) {
      els.peekFrame.value = String(state.snapshot.next_frame);
      els.peekFrameOut.textContent = `${els.peekFrame.value} / ${state.snapshot.frame_count}`;
      els.peekNote.textContent = "显示当前帧";
      renderFrameProbs(state.snapshot.frame_info);
    }
  });

  els.runBtn.addEventListener("click", runContinuously);
  els.stopBtn.addEventListener("click", () => {
    state.runAbort = true;
    state.running = false;
    updateButtons();
    updateStatusFromSnapshot();
  });

  els.restartBtn.addEventListener("click", async () => {
    // Reset by opening a fresh server session from the same sample or files.
    state.runAbort = true;
    if (state.loadedSample) {
      await loadSample(state.loadedSample);
      return;
    }
    if (!els.vocabFile.files[0] || !els.framesFile.files[0]) {
      showError("要回到起点请重新选择文件或重新载入样例");
      return;
    }
    const form = new FormData();
    form.append("vocab_file", els.vocabFile.files[0]);
    form.append("frames_file", els.framesFile.files[0]);
    form.append("beam_width", els.beamWidth.value);
    await createSession(form);
  });

  els.peekFrame.addEventListener("input", () => {
    const frame = Number(els.peekFrame.value);
    els.peekFrameOut.textContent = `${frame} / ${els.peekFrame.max}`;
    peek(frame);
  });

  async function loadSamples() {
    try {
      const payload = await api("/api/samples");
      state.samples = payload.samples || [];
      els.sampleButtons.innerHTML = "";
      if (!state.samples.length) {
        const span = document.createElement("span");
        span.className = "hint";
        span.textContent = "未找到样例文件";
        els.sampleButtons.appendChild(span);
        return;
      }
      state.samples.forEach((sample) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "sample ghost";
        btn.textContent = sample.name;
        btn.addEventListener("click", () => loadSample(sample.name));
        els.sampleButtons.appendChild(btn);
      });
    } catch { /* samples are optional */ }
  }

  async function loadSample(name) {
    els.setupError.hidden = true;
    setBusy(true, "载入样例…");
    try {
      const sample = await api(`/api/samples/${encodeURIComponent(name)}`);
      const payload = await api("/api/sessions", {
        method: "POST",
        body: JSON.stringify({
          vocab: sample.vocab,
          beam_width: Number(els.beamWidth.value),
          frames: sample.frames,
        }),
      });
      startSession(payload, name);
      showError("");
      hideNotice();
      showNotice(`已载入样例「${name}」，共 ${payload.frame_count} 帧。`);
    } catch (error) {
      els.setupError.hidden = false;
      els.setupError.textContent = error.message;
      setConnection("error", "加载失败");
    } finally {
      setBusy(false);
    }
  }

  loadSamples();
  updateButtons();
  updateStatusFromSnapshot();
})();
