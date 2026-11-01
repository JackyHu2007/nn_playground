/**
 * WebSocket client. Auto-reconnects with exponential backoff; retains the
 * last session id so a reconnecting client rejoins the same training run.
 */
export class Api {
  constructor() {
    this.ws = null;
    this.sessionId = null;
    this.handlers = {
      welcome: [], frame: [], error: [], open: [], close: [], node_boundary: [],
    };
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 15000;
    this.shouldReconnect = true;
    this.wsUrl = this._buildUrl();
  }

  _buildUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const base = `${proto}//${location.host}/ws`;
    return this.sessionId ? `${base}?session=${this.sessionId}` : base;
  }

  connect() {
    this.wsUrl = this._buildUrl();
    this.ws = new WebSocket(this.wsUrl);

    this.ws.onopen = () => {
      this.reconnectDelay = 1000;
      this._emit("open");
    };

    this.ws.onmessage = (evt) => {
      let data;
      try { data = JSON.parse(evt.data); } catch (e) {
        console.error("Bad JSON from server", evt.data);
        return;
      }
      if (data.type === "welcome") {
        this.sessionId = data.sessionId;
        this._emit("welcome", data);
      } else if (data.type === "error") {
        this._emit("error", data);
      } else if (data.type === "pong") {
        // no-op
      } else if (data.type === "node_boundary") {
        this._emit("node_boundary", data);
      } else {
        this._emit("frame", data);
      }
    };

    this.ws.onclose = () => {
      this._emit("close");
      if (!this.shouldReconnect) return;
      const delay = this.reconnectDelay;
      this.reconnectDelay = Math.min(
        this.reconnectDelay * 1.6, this.maxReconnectDelay
      );
      setTimeout(() => this.connect(), delay);
    };

    this.ws.onerror = (e) => {
      console.error("WS error", e);
    };
  }

  on(type, cb) {
    if (!this.handlers[type]) throw new Error(`Unknown event ${type}`);
    this.handlers[type].push(cb);
  }

  _emit(type, payload) {
    for (const cb of this.handlers[type] || []) cb(payload);
  }

  send(cmd) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn("WS not open, dropping command", cmd);
      return;
    }
    this.ws.send(JSON.stringify(cmd));
  }

  close() {
    this.shouldReconnect = false;
    if (this.ws) this.ws.close();
  }
}