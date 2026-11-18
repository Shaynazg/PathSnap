from datetime import datetime, timezone
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from .schemas import LogEvent
from . import storage

app = FastAPI(title="PathSnap Logging Receiver")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post('/ingest')
def ingest(event: LogEvent):
    storage.append(event.model_dump(mode='json'))
    return {"success": True, "event_id": event.event_id, "received_timestamp": datetime.now(timezone.utc).isoformat()}

@app.get('/logs')
def logs(severity: str | None = None, source: str | None = None, event_type: str | None = None, trip_id: str | None = None, limit: int = Query(100, ge=1, le=1000)):
    events = storage.recent(locals())
    return {"events": events, "total": len(events)}

@app.get('/health')
def health():
    return {"status":"healthy", "storage":"available", "file":str(storage.PATH)}

@app.get('/', response_class=HTMLResponse)
def dashboard():
    return '''<!doctype html><title>PathSnap Logs</title><h1>PathSnap Log Monitor</h1><button onclick="load()">Refresh</button><div id="stats"></div><table border="1"><thead><tr><th>Timestamp</th><th>Severity</th><th>Source</th><th>Event</th><th>Trip</th><th>Message</th></tr></thead><tbody id="rows"></tbody></table><script>async function load(){let e=(await(await fetch('/logs?limit=500')).json()).events;document.querySelector('#stats').textContent=`Total ${e.length} · Frontend ${e.filter(x=>x.source==='frontend').length} · Backend ${e.filter(x=>x.source==='backend').length} · Errors ${e.filter(x=>x.severity==='ERROR').length}`;document.querySelector('#rows').innerHTML=e.map(x=>`<tr>${[x.timestamp,x.severity,x.source,x.event_type,x.trip_id||'',x.message].map(v=>`<td>${String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}</td>`).join('')}</tr>`).join('')}load();setInterval(load,5000)</script>'''
