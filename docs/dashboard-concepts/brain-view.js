// The 3D renderer is lazy-loaded; the existing Circuit and inspector share the same state.
let brainViewer, brainLoading, brainError;
function brainSnapshot() {
  if(state.page==='live')return LiveWorkspace.brainSnapshot();
  return { selected:state.selected, region:state.region, frame:state.frame,
    heat:current().heat, activity:DemoTelemetry.activity(state.scenario,state.frame),
    surface:state.brainSurface, showEdges:$('#show-edges').checked };
}
function brainHover(node, event) {
  const tip=$('.tooltip');
  if(!node){tip.hidden=true;return;}
  tip.textContent=`${node.label} · ${regions[node.g].simple} · click to inspect`;
  tip.hidden=false;
  tip.style.left=Math.max(6,Math.min(event.clientX+13,innerWidth-tip.offsetWidth-10))+'px';
  tip.style.top=Math.min(event.clientY+15,innerHeight-45)+'px';
}
function brainFallback(message) {
  const host=$('#graph-stage');
  $('#neural-workspace').dataset.brainFallback='true';
  $('#network-visual-title').textContent='3D view unavailable';
  $('#network-visual-description').textContent='The Circuit and its inspector remain interactive.';
  $('#graph-view-provenance').textContent='STATIC AI ILLUSTRATION · NO WEBGL';
  $$('[data-action^="zoom"]').forEach(b=>b.disabled=true);
  $('#show-brain-surface').disabled=true;
  stop();
  const box=document.createElement('div');box.className='brain-fallback';
  const img=document.createElement('img');img.src='assets/brain-observatory.png';img.width=1254;img.height=1254;img.alt='Static illustrated brain fallback, not the interactive 3D model';
  const text=document.createElement('p');text.textContent=message;
  const button=document.createElement('button');button.className='btn';button.dataset.graphView='circuit';button.textContent='Explore the Circuit';
  box.append(img,text,button);host.replaceChildren(box);
}
function buildBrainGraph() {
  const live=state.page==='live';
  const host=$(live?'#live-graph-stage':'#graph-stage'),w=host.clientWidth,h=host.clientHeight;
  if(!w||!h)return;
  state.graph={mode:'brain',w,h};
  if(brainViewer){brainViewer.mount(host);brainViewer.update(brainSnapshot());return;}
  if(brainError){if(live)host.textContent=brainError;else brainFallback(brainError);return;}
  if(new URLSearchParams(location.search).has('WEBGL_OFF')){
    brainError='3D is disabled. This is a static illustration; the Circuit remains interactive.';
    if(live)host.textContent=brainError;else brainFallback(brainError);return;
  }
  if(brainLoading)return;
  host.innerHTML='<div class="brain-loading" role="status"><span>Loading the neural atlas…</span><small>Preparing 3D anatomy and sample connections</small></div>';
  brainLoading=import('./brain-3d/viewer.js?v=atlas-4').then(m=>m.createBrainViewer(nodes,edges,id=>state.page==='live'?LiveWorkspace.select(id):selectNeuron(id),brainHover)).then(viewer=>{
    brainViewer=viewer;
    if(state.page==='live'||state.graphMode==='brain'&&['overview','network'].includes(state.page))buildBrainGraph();
  }).catch(error=>{
    console.error('Brain viewer initialization:',error);
    brainError='The 3D view could not load. Reload to retry, or explore the interactive Circuit.';
    if(state.page==='live')host.textContent=brainError;else if(state.graphMode==='brain')brainFallback(brainError);
  });
}
function paintBrainActivity(){brainViewer?.update(brainSnapshot());}
function highlightBrainSelection(){brainViewer?.update(brainSnapshot());}
function tickBrain(progress){if(state.graphMode==='brain')brainViewer?.tick(state.frame+progress);}
function zoomBrain(action){brainViewer?.zoom(action==='zoom-in'?.15:-.15,action==='zoom-reset');}
function buildGraph() {
  if(state.graphMode==='brain')buildBrainGraph();
  else{brainViewer?.suspend();buildCircuitGraph();}
}
function setGraphMode(mode) {
  if(!['brain','circuit'].includes(mode))return;
  state.graphMode=mode;state.zoom=1;
  renderGraphControls();
  $('.tooltip').hidden=true;buildGraph();renderPlay();syncViewURL();
}
function renderGraphControls() {
  const mode=state.graphMode;
  delete $('#neural-workspace').dataset.brainFallback;
  $$('[data-action^="zoom"]').forEach(b=>b.disabled=false);
  $('#show-brain-surface').disabled=false;
  $('#neural-workspace').dataset.graphMode=mode;
  $$('[data-graph-view]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.graphView===mode));
  $('#graph-view-provenance').textContent=mode==='brain'?'3D ANATOMY · SYNTHETIC SNN':'SCHEMATIC POSITIONS · SAMPLE EDGES';
  $('#network-visual-title').textContent=mode==='brain'?'Inside a living network':'The neural circuit';
  $('#network-visual-description').textContent=mode==='brain'?'Turn it. Trace a connection. Discover a pattern.':'Four model regions and their sample connections.';
}
document.addEventListener('click',e=>{
  const control=e.target.closest('[data-graph-view]');
  if(control)setGraphMode(control.dataset.graphView);
});
document.addEventListener('change',e=>{
  if(e.target.id!=='show-brain-surface')return;
  state.brainSurface=e.target.checked;
  if(state.graph?.mode==='brain')paintBrainActivity();
  syncViewURL();
});
window.addEventListener('pagehide',e=>{if(!e.persisted)brainViewer?.dispose();});
