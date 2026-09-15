'use strict';
const $=(s,r=document)=>r.querySelector(s),$$=(s,r=document)=>[...r.querySelectorAll(s)],NS='http://www.w3.org/2000/svg';
const paths={grid:'M3 3h7v7H3V3m11 0h7v7h-7V3M3 14h7v7H3v-7m11 0h7v7h-7v-7',network:'M4 7l8-4 8 5-2 10-11 2-4-8 9-9 6 15-14-11m3 13L20 8M12 3v12M3 12l9 3 8-7','circle-dot':'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0m-7 0a2 2 0 1 1-4 0 2 2 0 0 1 4 0',journal:'M6 3h14v18H6a3 3 0 0 1 0-6h14M6 3a3 3 0 0 0-3 3v12m6-11h7m-7 4h5',book:'M12 5c-3-2-6-2-9-1v15c3-1 6-1 9 1m0-15c3-2 6-2 9-1v15c-3-1-6-1-9 1V5',download:'M12 3v12m-5-5 5 5 5-5M4 17v4h16v-4',code:'m8 6-6 6 6 6m8-12 6 6-6 6M14 3l-4 18',info:'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0m-9-1v6m0-10v.1','arrow-up-right':'M6 18 18 6M6 6h12v12','arrow-left':'M19 12H5m6-6-6 6 6 6',chevron:'m6 9 6 6 6-6',check:'m5 12 4 4L19 6',plus:'M12 5v14M5 12h14',minus:'M5 12h14',maximize:'M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5',spark:'m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3',keyboard:'M3 6h18v12H3V6m4 4h.1M11 10h.1M15 10h.1M18 10h.1M7 14h10',window:'M3 4h18v16H3V4m0 5h18M7 6.5h.1m3 0h.1',audio:'M4 10v4m4-8v12m4-15v18m4-15v12m4-8v4',moon:'M20 15A9 9 0 0 1 9 4a9 9 0 1 0 11 11',activity:'M2 12h5l3-8 4 16 3-8h5',play:'m8 5 11 7-11 7V5',pause:'M8 5v14m8-14v14',x:'m6 6 12 12M18 6 6 18',clock:'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0m-9-5v5l3 2'};
const icon=n=>`<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="${paths[n]||paths.info}"/></svg>`;
function hydrate(root=document){$$('[data-icon]',root).forEach(e=>e.outerHTML=icon(e.dataset.icon));$$('[data-logo]',root).forEach(e=>e.innerHTML='<svg class="brand-logo" viewBox="0 0 36 36" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="1.2"><path d="m8 11 10-6 11 7-1 13-10 6-11-8 1-12m10-6v13l11-6m-22 11 11-5 10 7m-10-7v13M8 11l10 7"/></g><g fill="currentColor"><circle cx="18" cy="5" r="2.2"/><circle cx="8" cy="11" r="2.2"/><circle cx="29" cy="12" r="2.2"/><circle cx="7" cy="23" r="2.2"/><circle cx="18" cy="18" r="2.5"/><circle cx="28" cy="25" r="2.2"/><circle cx="18" cy="31" r="2.2"/></g></svg>')}
function svgEl(name,attrs={}){const e=document.createElementNS(NS,name);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);return e}
const safe=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const reduced=matchMedia('(prefers-reduced-motion: reduce)'),current=()=>scenarios[state.scenario],currentLabel=()=>state.learned[slotKey()]||current().label;
function setText(selector,value){$$(selector).forEach(e=>{if(e.textContent!==value)e.textContent=value})}
function renderRecognition() {
  const s=current(), label=currentLabel(), learned=!!state.learned[slotKey()];
  $('#state-banner').classList.toggle('unknown',!label);
  setText('[data-current-label]',label || 'Not yet named');
  setText('[data-state-prefix]',learned?'You annotated ':label?'A pattern called ':'');
  setText('[data-state-description]',learned?'Your annotation is bound to this context and frame. No later recognition has been measured.':s.description);
  setText('[data-state-kicker]',learned?'Local annotation / this exact frame':label?'Example recognition / user-defined vocabulary':'No example match / open for annotation');
  setText('[data-state-source]',learned?'Tab-only annotation · export to retain':'Authored example score · not calibrated confidence');
  setText('[data-match-value]',learned?'—':s.match===null?'—':s.match.toFixed(2));
  setText('[data-match-caption]',learned?'not evaluated':s.match===null?'no match':'example score');
  setText('[data-feedback-title]',label?'Does this label fit?':'Keep room for the unknown.');
  setText('[data-feedback-copy]',label?'Confirm or add your own word. The annotation stays with this exact moment.':'Leave it unnamed, or attach a word to this moment. There is no need to force a match.');
  $$('[data-action=confirm]').forEach(b=>{b.disabled=!label||state.confirmed.has(slotKey())});
  setText('[data-confirm-text]',state.confirmed.has(slotKey())?'Annotated':!label?'No label to confirm':'Confirm label');
}
function renderState(){
  const s=current();
  renderRecognition();
  $('[data-signal-chips]').innerHTML=s.signals.map((t,i)=>'<span>'+icon(['keyboard','window','activity'][i])+safe(t)+'</span>').join('');
  const extras = [
    ['activity','Mouse activity','Example pointer rhythm',state.scenario==='rest'?'Low':'Intermittent'],
    ['moon','Idle time','Example inactivity context',state.scenario==='rest'?'Extended':'Brief'],
    ['clock','Time of day','Illustrated context time',s.time]
  ];
  $('[data-sensor-rows]').innerHTML=[...s.sensors,...extras].map(([i,name,copy,value])=>
    '<div class="sensor-row"><div class="sensor-icon">'+icon(i)+'</div><div><strong>'+name+'</strong><p>'+copy+'</p></div><span class="sensor-value">'+value+'</span></div>').join('');
  $('#modulator-grid').innerHTML=modDefs.map((m,i)=>'<button class="panel modulator" data-modulator="'+i+'" style="--series:var(--'+m.color+')" aria-label="Explain '+m.name+'"><span class="modulator-top"><span>'+m.code+' / '+m.name+'</span>'+icon('info')+'</span><h3>'+m.title+'</h3><span class="modulator-value">'+s.mods[i].toFixed(2)+'<small>/ 1.00</small></span><div class="meter" role="meter" aria-label="'+m.name+', synthetic level" aria-valuemin="0" aria-valuemax="1" aria-valuenow="'+s.mods[i]+'"><span style="width:'+s.mods[i]*100+'%"></span></div><p>'+m.desc+'</p></button>').join('');
  renderMeaning();renderVocabulary();renderRecent();renderSession();
  if(state.graph)paintActivity();
  if(state.page==='network')drawRaster();
}
function renderMeaning(){const r=regions.find(r=>r.key===state.region),s=current(),label=currentLabel();$('#meaning-title').textContent=r?r.meaning:state.learned[slotKey()]?'A word for this moment.':label?'Why it looks familiar.':'A question, not a guess.';$('#meaning-intro').textContent=r?r.description:'Three parts of the story come together.';const steps=r?[[r.simple+' region',`${r.n} ${r.key==='e'?'projection units':'neurons'} in the default model.`],['Its role in learning',r.technical],['Keep the distinction','A region describes computation. Your felt-state vocabulary describes a learned signature.']]:[['The desktop',s.sensors[0][2]+'. '+s.sensors[1][2]+'.'],['The model',state.learned[slotKey()]?'This frozen signature has just been given a label. No new recognition has been measured.':s.reason],['Your words',label?`“${label}” is a user-defined label, not a built-in emotion.`:'An unfamiliar signature stays unnamed until you decide what to call it.']];$('#reason-list').innerHTML=steps.map(([a,b],i)=>`<div class="reason"><span class="reason-num">${i+1}</span><div><strong>${safe(a)}</strong><p>${safe(b)}</p></div></div>`).join('');$('#graph-story').textContent=r?r.description:s.story;}
function setInsight(tab){state.insight=tab;$$('[data-insight]').forEach(b=>{const active=b.dataset.insight===tab;b.setAttribute('aria-selected',active);b.tabIndex=active?0:-1});$('#meaning-pane').hidden=tab!=='meaning';$('#neuron-pane').hidden=tab!=='neuron';}
function graphPoint(n,w,h){const rnd=random(9381+n.g*3901+n.index*171),a=rnd()*Math.PI*2,r=Math.sqrt(rnd()),small=w<540;const centers=small?[[.15,.48],[.5,.42],[.83,.58],[.5,.81]]:[[.13,.47],[.4,.45],[.72,.42],[.86,.82]],radius=(small?[.09,.175,.105,.071]:[.085,.137,.09,.055])[n.g]*w;return{x:centers[n.g][0]*w+Math.cos(a)*r*radius,y:centers[n.g][1]*h+Math.sin(a)*r*radius*.8};}
function buildCircuitGraph(){const host=$('#graph-stage'),w=host.clientWidth,h=host.clientHeight;if(!w||!h)return;const svg=svgEl('svg',{viewBox:`0 0 ${w} ${h}`,class:'graph-svg',role:'img','aria-label':'Schematic network with 1,000 selectable units. Input 200, Expansion 500, Concepts 200, Memory 100. For keyboard selection, use the Single neuron tab.'}),scene=svgEl('g'),grid=svgEl('g',{stroke:'var(--line)','stroke-width':'.45',opacity:'.4'}),positions=new Map(nodes.map(n=>[n.id,graphPoint(n,w,h)]));for(let x=18;x<w;x+=32)grid.append(svgEl('path',{d:`M${x} 20V${h-20}`}));for(let y=20;y<h-20;y+=32)grid.append(svgEl('path',{d:`M18 ${y}H${w-18}`}));svg.append(grid);const edgeLayer=svgEl('g',{'data-edges':'',opacity:'.2','pointer-events':'none'});edges.filter((_,i)=>i%19===0).forEach(e=>{const a=positions.get(e.from),b=positions.get(e.to);edgeLayer.append(svgEl('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:`var(--${e.to[0]})`,'stroke-width':'.7'}))});scene.append(edgeLayer);const marks=svgEl('g',{'data-neurons':''});nodes.forEach(n=>{const p=positions.get(n.id);marks.append(svgEl('circle',{cx:p.x,cy:p.y,r:1.3,fill:`var(--${n.key})`,class:'neuron','data-id':n.id}))});scene.append(marks);const selection=svgEl('g',{'data-selection':'','pointer-events':'none'});scene.append(selection);svg.append(scene);const labels=svgEl('g',{'pointer-events':'none'});regions.forEach((r,g)=>{const small=w<540,x=(small?[.17,.55,.8,.47]:[.13,.4,.72,.86])[g]*w,y=(small?[.22,.13,.8,.98]:[.2,.13,.18,.99])[g]*h;const label=svgEl('text',{x,y:Math.min(h-18,y),'text-anchor':'middle',class:'region-title'});const title=svgEl('tspan',{x,dy:0,class:'region-name'});title.textContent=`${r.simple} · ${r.n}`;label.append(title);const caption=svgEl('tspan',{x,dy:15,class:'region-caption'});caption.textContent=r.verb;label.append(caption);labels.append(label)});svg.append(labels);host.replaceChildren(svg);state.graph={svg,scene,positions,marks,edgeLayer,selection,w,h};applyZoom();paintActivity();highlightSelection();svg.addEventListener('click',e=>{const node=e.target.closest('[data-id]');if(node)selectNeuron(node.dataset.id)});svg.addEventListener('pointermove',e=>{const mark=e.target.closest('[data-id]'),tip=$('.tooltip');if(!mark){tip.hidden=true;return}const n=byId.get(mark.dataset.id);tip.textContent=`${n.label} · ${regions[n.g].simple} · click to inspect`;tip.hidden=false;tip.style.left=Math.max(6,Math.min(e.clientX+13,innerWidth-tip.offsetWidth-10))+'px';tip.style.top=Math.min(e.clientY+15,innerHeight-45)+'px'});svg.addEventListener('pointerleave',()=>$('.tooltip').hidden=true);}
function paintActivity(){const g=state.graph;if(!g)return;if(g.mode==='brain'){paintBrainActivity();return;}const s=current();for(const dot of g.marks.children){const n=byId.get(dot.dataset.id),dim=state.region!=='all'&&state.region!==n.key,on=DemoTelemetry.sample(state.scenario,n.id,state.frame).glow>0;dot.setAttribute('opacity',dim?'.08':on?'.96':'.42');dot.setAttribute('r',on?'2.05':'1.2')}g.edgeLayer.style.display=$('#show-edges').checked?'':'none';g.edgeLayer.setAttribute('opacity',state.region==='all'?'.2':'.07');}
function highlightSelection(){const g=state.graph;if(!g)return;if(g.mode==='brain'){highlightBrainSelection();return;}const n=byId.get(state.selected),p=g.positions.get(n.id);g.selection.replaceChildren();const related=edges.filter(e=>e.from===n.id||e.to===n.id);if($('#show-edges').checked)related.forEach(e=>{const a=g.positions.get(e.from),b=g.positions.get(e.to);g.selection.append(svgEl('line',{x1:a.x,y1:a.y,x2:b.x,y2:b.y,stroke:`var(--${n.key})`,'stroke-width':'1',opacity:'.75'}));const other=g.positions.get(e.from===n.id?e.to:e.from);g.selection.append(svgEl('circle',{cx:other.x,cy:other.y,r:3,fill:`var(--${(e.from===n.id?e.to:e.from)[0]})`,opacity:'.9'}))});g.selection.append(svgEl('circle',{cx:p.x,cy:p.y,r:7,fill:'none',stroke:'var(--ink)','stroke-width':'1'}));const right=p.x>g.w-77;const t=svgEl('text',{x:p.x+(right?-13:13),y:p.y-11,'text-anchor':right?'end':'start',class:'node-label'});t.textContent=n.label;g.selection.append(t);}
function applyZoom(){const g=state.graph;if(g?.mode==='brain')return;if(g)g.scene.setAttribute('transform',`translate(${g.w*.5*(1-state.zoom)} ${g.h*.5*(1-state.zoom)}) scale(${state.zoom})`)}
function traceValues(n){return DemoTelemetry.series(state.scenario,n.id).voltage}
function renderNeuron(){
  const n=byId.get(state.selected),r=regions[n.g];
  $('#neuron-name').textContent=n.label;
  $('#neuron-region').textContent=r.name+' · '+(n.key==='e'?'fixed projection':'spiking layer');
  $('#neuron-event-name').textContent=n.key==='e'?'Active samples to cursor':'Spikes to cursor';
  $('#neuron-metric-name').textContent=n.key==='e'?'Output at cursor':'Example membrane';
  $('#neuron-unit').textContent=n.key==='e'?'binary':'a.u.';
  $('#trace-label').textContent=n.key==='e'?'Binary projection fixture':'Procedural membrane trace';
  $('#neuron-description').textContent=r.technical;
  $('#neuron-region-select').value=n.key;$('#neuron-index').max=r.n-1;$('#neuron-index').value=n.index;
  const svg=$('#neuron-trace');svg.setAttribute('viewBox','0 0 280 76');
  svg.setAttribute('aria-label',n.label+': procedural '+(n.key==='e'?'binary output':'normalized membrane')+' for 200 samples. Not recorded telemetry.');
  svg.replaceChildren();
  [10,36,62].forEach(y=>svg.append(svgEl('path',{d:'M0 '+y+'H280',stroke:'var(--line)','stroke-width':'.6'})));
  const values=traceValues(n),d=Array.from(values,(v,i)=>(i?'L':'M')+(i/199*280).toFixed(2)+' '+(64-v*53).toFixed(2)).join(' ');
  svg.append(svgEl('path',{d,fill:'none',stroke:'var(--'+n.key+')','stroke-width':'1.3'}));
  svg.append(svgEl('path',{id:'trace-cursor',stroke:'var(--ink)','stroke-width':1,'stroke-dasharray':'2 3'}));
  svg.append(svgEl('circle',{id:'trace-dot',r:3,fill:'var(--ink)'}));
  renderConnections(n);renderSampleFrame();
}
function renderConnections(n){
  const incoming=edges.filter(e=>e.to===n.id),outgoing=edges.filter(e=>e.from===n.id);
  const related=[...incoming.map(e=>e.from),...outgoing.map(e=>e.to)];
  const links=limit=>related.slice(0,limit).map(id=>'<button class="neighbor" data-neighbor="'+id+'" aria-label="Inspect connected unit '+byId.get(id).label+'">'+byId.get(id).label+'</button>').join('');
  $('#neuron-neighbors').innerHTML=links(5);
  const first=incoming[0]||outgoing[0];
  $('#connection-detail').innerHTML='<div class="row between wrap" style="margin:18px 0"><span class="mono">'+n.label+'</span><span class="pill">'+incoming.length+' in / '+outgoing.length+' out · sample</span></div>'+
    (first?'<p class="plot-caption"><strong>'+byId.get(first.from).label+' → '+byId.get(first.to).label+'</strong></p><dl class="provenance-list"><div><dt>Pathway</dt><dd>'+(first.plastic?'Plastic in source model':'Fixed projection')+'</dd></div><div><dt>Sample weight</dt><dd class="mono">'+first.weight.toFixed(2)+' · generated</dd></div><div><dt>Measured change</dt><dd>Not available</dd></div></dl>':'<p class="plot-caption">No adjacent edges in this sample.</p>')+
    '<p class="plot-caption">Seeded example edges, not a checkpoint’s weight matrices. Follow a connected unit below; the 3D selection uses the same catalog.</p><div class="neuron-neighbors">'+links(8)+'</div>';
}
function renderSampleFrame(){
  const n=byId.get(state.selected),sample=DemoTelemetry.sample(state.scenario,n.id,state.frame);
  setText('#neuron-spikes',String(sample.count));
  setText('#neuron-metric',n.key==='e'?String(sample.output):sample.voltage.toFixed(3));
  setText('#trace-window','t +'+(state.frame/100).toFixed(3)+' s');
  $('#neuron-pane').dataset.frame=String(state.frame);
  $('#neuron-pane').dataset.output=String(sample.output);
  $('#neuron-spikes').title=sample.total+' total events in the full 2 s example window';
  const x=state.frame/199*280;
  $('#trace-cursor')?.setAttribute('d','M'+x+' 5V70');
  $('#trace-dot')?.setAttribute('cx',x);
  $('#trace-dot')?.setAttribute('cy',64-sample.voltage*53);
  const cursor=$('#raster-cursor');
  if(cursor){const left=Number(cursor.dataset.left),width=Number(cursor.dataset.width);cursor.setAttribute('d','M'+(left+state.frame/200*width)+' 22V120')}
  const active=nodes.reduce((count,node)=>count+DemoTelemetry.sample(state.scenario,node.id,state.frame).output,0);
  setText('#sample-active',String(active));
}
function selectNeuron(id){state.selected=id;const n=byId.get(id);if(state.region!=='all'&&state.region!==n.key)setRegion(n.key,false);renderNeuron();setInsight('neuron');highlightSelection();if(state.page==='network')drawRaster();syncViewURL()}
function setRegion(region,meaning=true){state.region=region;$$('button[data-region]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.region===region));if(region!=='all'&&byId.get(state.selected).key!==region){state.selected=region+Math.min(42,regions.find(r=>r.key===region).n-1);renderNeuron()}renderMeaning();paintActivity();highlightSelection();if(meaning)setInsight('meaning');syncViewURL();}
function drawRaster(){
  const svg=$('#spike-raster'),w=svg.clientWidth;if(!w)return;
  svg.setAttribute('viewBox','0 0 '+w+' 148');svg.replaceChildren();
  const left=37,top=24,width=w-50,height=91,selected=byId.get(state.selected);
  [0,20,39].forEach(n=>{const y=top+(39-n)/39*height;svg.append(svgEl('path',{d:'M'+left+' '+y+'H'+(left+width),class:'grid-line'}));svgText(svg,27,y+3,n,'end')});
  svgText(svg,left,12,'Concept ID','start');
  [0,1,2].forEach(t=>svgText(svg,left+t/2*width,133,t));
  svgText(svg,left+width/2,147,'Time (s) · same fixture');
  for(let n=0;n<40;n++){
    const y=top+(39-n)/39*height,series=DemoTelemetry.series(state.scenario,'c'+n);
    if(selected.key==='c'&&selected.index===n)svg.append(svgEl('path',{d:'M'+left+' '+y+'H'+(left+width),stroke:'var(--w)','stroke-width':'3',opacity:'.3'}));
    const row=svgEl('g',{'data-raster-unit':'c'+n});
    row.append(svgEl('rect',{x:left,y:y-1.2,width,height:2.4,fill:'transparent'}));
    series.output.forEach((output,frame)=>{if(output){const x=left+frame/200*width;row.append(svgEl('line',{x1:x,y1:y-1,x2:x,y2:y+1,stroke:selected.id==='c'+n?'var(--w)':'var(--mint)','stroke-width':'1',opacity:'.85'}))}});
    svg.append(row);
  }
  svg.append(svgEl('path',{id:'raster-cursor','data-left':left,'data-width':width,stroke:'var(--ink)','stroke-width':1,'stroke-dasharray':'3 3','pointer-events':'none'}));
  renderSampleFrame();
}
function svgText(svg,x,y,text,anchor='middle'){const t=svgEl('text',{x,y,'text-anchor':anchor});t.textContent=text;svg.append(t)}
function setFrame(frame){
  const previousLabel=currentLabel(),wasAnnotated=!!state.learned[slotKey()];
  state.frame=Math.max(0,Math.min(199,Math.round(frame)));
  $('#frame-slider').value=state.frame;
  $('#frame-slider').setAttribute('aria-valuetext','Frame '+state.frame+', '+(state.frame/100).toFixed(2)+' seconds');
  setText('#frame-number',String(state.frame).padStart(3,'0'));
  setText('#replay-time','t +'+(state.frame/100).toFixed(3)+' s');
  paintActivity();renderSampleFrame();renderRecognition();
  if(previousLabel!==currentLabel()||wasAnnotated!==!!state.learned[slotKey()])renderMeaning();
}
function stop(){state.playing=false;cancelAnimationFrame(state.raf);renderPlay();syncViewURL()}
function renderPlay(){
  const available=['overview','network'].includes(state.page)&&!(state.graphMode==='brain'&&brainError);
  $('#frame-slider').disabled=!available;$('#replay-speed').disabled=!available;
  $$('[data-action=play]').forEach(b=>{b.disabled=!available;b.innerHTML=icon(state.playing?'pause':'play');b.setAttribute('aria-pressed',state.playing);b.setAttribute('aria-label',state.playing?'Pause simulated neural activity':'Play simulated neural activity')});
  setText('[data-playback-label]',!available?'Replay unavailable':state.playing?'Replaying fixture':state.frame===199?'Replay complete':'Replay paused');
}
function play(){
  if(state.playing){stop();return}
  if(!['overview','network'].includes(state.page))return;
  if(reduced.matches){notify('Reduced motion is enabled. Use the frame slider to inspect each sample.');return}
  if(state.frame===199)setFrame(0);
  state.playing=true;state.last=0;renderPlay();state.raf=requestAnimationFrame(animate);
}
function animate(t){
  if(!state.playing)return;
  if(!state.last)state.last=t;
  if(t-state.last>=100){
    const step=Math.max(1,Math.round((t-state.last)*state.speed/10));
    setFrame(state.frame+step);state.last=t;
    if(state.frame===199){stop();return}
  }
  tickBrain(Math.min(10,(t-state.last)*state.speed/10));
  state.raf=requestAnimationFrame(animate);
}
function changeMoment(key,momentIndex){
  stop();state.scenario=key;state.eventId=null;
  state.moment=momentIndex??({focus:5,switching:2,rest:3,unknown:4}[key]);
  $('#moment-select').value=key;
  renderState();renderNeuron();renderJournal();setFrame(state.frame);syncViewURL();
}
const pageCopy={
  live:['Live session','Local observation / opt-in inputs','See what actually happens.','Actual model steps. Explicit input sources. No inferred feelings or generated telemetry.'],
  overview:['Observatory','01 / Observe the system','Understand the state.','From input patterns to learned meaning. Every interpretation stays open to correction.'],
  network:['Neural explorer','02 / Inspect the mechanism','Follow the signal.','Explore the anatomy. Inspect the computation. Know which is which.'],
  vocabulary:['Vocabulary','03 / Define the meaning','A language, learned together.','Your annotations give recurring patterns meaning — without prescribing how you feel.'],
  journal:['Learning journal','04 / Trace the evidence','Keep the moment. Keep the context.','A reviewable record of example observations and your own annotations.'],
  methods:['Methods & provenance','05 / Establish the evidence','Clarity is part of the method.','Sources, assumptions and limits — visible alongside the work, not buried beneath it.']
};
function setPage(){
  const hash=location.hash.slice(1);
  state.page=Object.hasOwn(pageCopy,hash)?hash:'live';
  stop();document.body.dataset.page=state.page;
  $$('.screen').forEach(e=>e.classList.toggle('active',e.id===state.page));
  $$('[data-nav]').forEach(a=>{
    a.classList.toggle('active',a.dataset.nav===state.page);
    if(a.dataset.nav===state.page)a.setAttribute('aria-current','page');
    else a.removeAttribute('aria-current');
  });
  const p=pageCopy[state.page];
  ['[data-page-name]','[data-page-kicker]','[data-page-title]','[data-page-subtitle]'].forEach((key,i)=>setText(key,p[i]));
  if(state.page==='overview'||state.page==='network'){
    $(state.page==='overview'?'#overview-network-slot':'#explorer-network-slot').append($('#neural-workspace'));
    requestAnimationFrame(()=>{buildGraph();if(state.page==='network'){setInsight('neuron');drawRaster()}});
  }else{brainViewer?.suspend()}
  window.scrollTo({top:0,behavior:'instant'});$('.tooltip').hidden=true;
  document.title='brAIn — '+p[0];
  if(appReady)syncViewURL();
  if(typeof LiveWorkspace!=='undefined')LiveWorkspace.onPageChange();
}
function renderVocabulary(){
  const search=$('#vocab-search').value.trim().toLowerCase(),words=vocabulary.filter(v=>v.label.toLowerCase().includes(search));
  setText('[data-word-count]',String(vocabulary.length).padStart(2,'0'));
  setText('[data-vocab-results]',words.length+' label'+(words.length===1?'':'s'));
  $('#vocab-empty').hidden=words.length>0;$('#word-detail').hidden=words.length===0;
  if(words.length&&!words.some(v=>v.label===state.word))state.word=words[0].label;
  $('#vocab-grid').innerHTML=words.map(v=>'<button class="panel word-card '+(state.word===v.label?'active':'')+'" data-word="'+safe(v.label)+'" aria-pressed="'+(state.word===v.label)+'"><div class="eyebrow">'+(v.sessionAnnotations?v.origin==='Taught in this demo'?'Local label':'Example · locally refined':safe(v.origin))+'</div><h3>'+safe(v.label)+'</h3><p>'+safe(v.note)+'</p><div class="row"><span>'+(v.sessionAnnotations?v.sessionAnnotations+' local · ':'')+(v.corrections-(v.sessionAnnotations||0))+' example annotations'+'</span>'+icon('arrow-up-right')+'</div></button>').join('');
  renderWordDetail();
}
function renderWordDetail(){const v=vocabulary.find(v=>v.label===state.word)||vocabulary[0];$('#word-detail').innerHTML=`<div class="eyebrow">Prototype reference / illustrative</div><h2>${safe(v.label)}</h2><p>${safe(v.note)}</p><div class="word-signature">${modDefs.map((m,i)=>`<div class="signature-row"><span>${m.code}</span><div class="meter" style="--series:var(--${m.color})"><span style="width:${v.mods[i]*100}%"></span></div><span>${v.mods[i].toFixed(2)}</span></div>`).join('')}</div><p style="font-size:11px">An illustrative four-value view of the signature. Context values are a frozen fixture, not measured neurochemistry. The real model also uses trends and context.</p><div class="rule"></div><div class="definition-row"><span>Named by</span><strong style="font-weight:400">${v.origin==='Taught in this demo'?'This session':'Example contributor'}</strong></div><div class="definition-row"><span>Source</span><strong style="font-weight:400">${v.origin==='Taught in this demo'?'Local annotation':v.sessionAnnotations?'Example + local revision':'Synthetic example'}</strong></div><div class="definition-row"><span>Annotations</span><strong style="font-weight:400">${v.sessionAnnotations||0} local / ${v.corrections-(v.sessionAnnotations||0)} example</strong></div><button class="text-button" data-action="learning">What is a prototype? ${icon('arrow-up-right')}</button>`;}
function renderRecent(){const shown=events.slice(0,3);$('#recent-moments').innerHTML=shown.map(e=>`<div class="moment-row"><time>${e.time}</time><div><strong>${safe(e.title)}</strong><p>${safe(e.copy)}</p></div></div>`).join('')}
function renderJournal(){
  const visible=events.filter(e=>state.filter==='all'||e.type===state.filter);
  $('#journal-events').innerHTML=visible.map(e=>'<article class="journal-event" data-selected="'+(state.eventId===e.id)+'"><time>'+e.time+'</time><span class="event-symbol">'+icon(e.icon)+'</span><div><h3>'+safe(e.title)+'</h3><p>'+safe(e.copy)+'</p><div class="eyebrow">'+(e.local?'LOCAL ANNOTATION · '+e.id:'SYNTHETIC '+(e.type==='correction'?'ANNOTATION':'OBSERVATION'))+'</div></div><button class="text-button" data-event="'+e.id+'" aria-pressed="'+(state.eventId===e.id)+'" aria-label="Inspect '+safe(e.title)+'">Inspect '+icon('arrow-up-right')+'</button></article>').join('');
  $('#day-strip').innerHTML=moments.map((m,i)=>'<button class="day-segment" style="flex:'+m.width+'" data-moment="'+i+'" aria-pressed="'+(state.moment===i&&!state.eventId)+'" aria-label="'+m.time+', '+m.name+'"><span>'+m.name+'</span></button>').join('');
  renderJournalMoment();renderRecent();
}
function journalSnapshot(){
  const event=events.find(event=>event.id===state.eventId),m=moments[state.moment];
  return event?.snapshot || {scenario:m.scenario,frame:42,title:m.name,time:m.time,mods:[...scenarios[m.scenario].mods],moment:state.moment,selected:'c42',region:'all',graphMode:state.graphMode};
}
function renderJournalMoment(){
  const event=events.find(e=>e.id===state.eventId),snap=journalSnapshot(),m=moments[state.moment];
  $('#journal-moment').innerHTML='<div class="eyebrow">'+(event?.local?'Annotation snapshot / '+event.id:'Selected context / '+snap.time)+'</div><h2>'+safe(event?.label||snap.title)+'</h2><p>'+safe(event?.copy||m.copy)+'</p><dl class="provenance-list"><div><dt>Context</dt><dd>'+safe(scenarios[snap.scenario].title)+'</dd></div><div><dt>Frame</dt><dd class="mono">'+String(snap.frame).padStart(3,'0')+' / 199 · '+(snap.frame/100).toFixed(2)+' s</dd></div><div><dt>Unit</dt><dd class="mono">'+byId.get(snap.selected).label+'</dd></div></dl><div class="quote">'+(event?.local?'This is the exact snapshot you annotated. Subsequent context changes do not move the annotation.':'Illustrated history, not a desktop recording. An observation does not establish how someone felt.')+'</div>'+(event?.reference?'<p class="method-note"><strong>Annotation note</strong><br>'+safe(event.reference.note)+'</p>':'')+(event?.previous?'<p class="method-note">Previous reference: '+event.previous.mods.map(v=>v.toFixed(2)).join(' / ')+'<br>Snapshot reference: '+event.reference.mods.map(v=>v.toFixed(2)).join(' / ')+'<br><small>DA / NE / ACh / 5-HT · authored values</small></p>':'')+'<button class="btn primary" data-action="replay-moment">Inspect this snapshot '+icon('arrow-up-right')+'</button>';
}
function showInfo(html){stop();$('#info-content').innerHTML=html;const title=$('#info-content h2');if(title)title.id='info-title';$('#info-dialog').showModal()}
function why(){const s=current(),label=currentLabel();showInfo(`<div class="eyebrow">From signals to a word</div><h2>${label?'Why “'+safe(label)+'”?':'Why is this unfamiliar?'}</h2><p>${state.learned[slotKey()]?'You have just named this example signature. A new match can only be evaluated when a future observation arrives.':safe(s.reason)}</p><div class="explain-flow"><div class="explain-step"><span class="reason-num">1</span><div><h3>Observe a pattern</h3><p>${safe(s.signals.join(' · '))}. These are input patterns, not a psychological conclusion.</p></div></div><div class="explain-step"><span class="reason-num">2</span><div><h3>Build an internal signature</h3><p>The network’s activity and prediction-error-driven modulators form a changing internal state. Trends summarize how it develops.</p></div></div><div class="explain-step"><span class="reason-num">3</span><div><h3>Compare with what you taught it</h3><p>A separate felt-state model compares this signature with learned prototypes. The word is supplied by you, not by one concept neuron.</p></div></div><div class="explain-step"><span class="reason-num">4</span><div><h3>Keep room for correction</h3><p>A similarity score is not a calibrated probability. The model can be uncertain; your correction can give the right moment a better name.</p></div></div></div><div class="source-note">MODEL BASIS: brain/core.py → bridge/felt_state.py<br>THIS VIEW: synthetic signatures and examples; no live measurement.</div>`)}
function teach(){
  stop();$('#teach-form').reset();$('#teach-error').hidden=true;$('#new-word').removeAttribute('aria-invalid');
  state.frozen=freezeMoment();
  const snap=state.frozen;
  $('#frozen-moment').innerHTML='<strong>'+safe(snap.title)+'</strong> · '+snap.time+'<br>Frame '+String(snap.frame).padStart(3,'0')+' / 199 · t +'+(snap.frame/100).toFixed(3)+' s · '+byId.get(snap.selected).label+'<br>DA '+snap.mods[0].toFixed(2)+' / NE '+snap.mods[1].toFixed(2)+' / ACh '+snap.mods[2].toFixed(2)+' / 5-HT '+snap.mods[3].toFixed(2)+'<br>Procedural fixture · not a live measurement.';
  $('#teach-dialog').showModal();$('#new-word').focus();
}
function confirm(){
  const label=currentLabel();if(!label||state.confirmed.has(slotKey()))return;
  const snap=freezeMoment(),v=vocabulary.find(v=>v.label===label),previous=v?structuredClone(v):null;
  state.confirmed.add(slotKey());
  recordAnnotation('You confirmed “'+label+'”','Bound to '+snap.title.toLowerCase()+', frame '+String(snap.frame).padStart(3,'0')+'. No model weights were changed.',snap,label,previous);
  renderState();renderJournal();notify('Label confirmed for this frame. Export the annotation to retain it.');
}

function desktop(){showInfo(`<div class="eyebrow">Living Desktop Manager / interaction preview</div><h2>A quieter workspace, by invitation.</h2><p>When you confirm a focused state, the product could offer a small, reversible change to your desktop. It should explain the suggestion before asking you to accept it.</p><div class="frozen-moment"><strong>Proposed: protect a 25-minute work block.</strong><br>Keep the active workspace; defer non-urgent notifications; show what was deferred afterwards.</div><p>In this design study, accepting changes only the preview status. No applications, notifications or system settings are modified.</p><div class="dialog-actions"><button class="btn" data-action="close">Not now</button><button class="btn primary" data-action="quiet-preview">${state.quietPreview?'End the demo preview':'Try it in this demo'} ${icon('arrow-up-right')}</button></div>`)}
const info={about:`<div class="eyebrow">Observatory / research workspace 03</div><h2>Make the system legible.</h2><p>An open-source design study for brAIn: inspect the model, challenge an interpretation and preserve the context of every annotation.</p><h3>Five connected views</h3><p>Observatory establishes context. Neural explorer connects spatial inspection to individual units. Vocabulary keeps user-defined meaning visible. Learning journal preserves snapshots. Methods &amp; provenance states what is known, generated and still missing.</p><h3>A prototype, with explicit boundaries</h3><p>All activity is a deterministic browser fixture. There is no live experiment, sensor connection, trained checkpoint or operating-system control here.</p><div class="source-note">Local annotations are held in this tab only. Export to keep them.<br>Model basis: f8ba635 · dataset OBS-DEMO-03.</div>`,guide:`<div class="eyebrow">An instrument, made legible</div><h2>Start with the pattern.<br>Then follow the detail.</h2><div class="explain-flow"><div class="explain-step"><span class="reason-num">1</span><div><h3>Read the recognized state</h3><p>The large word is a user-taught label. Its match score describes model similarity, not certainty about a feeling.</p></div></div><div class="explain-step"><span class="reason-num">2</span><div><h3>Explore the landscape</h3><p>Brain and Circuit share the same four model colors. Size and opacity show activity. Atlas labels name the source anatomy, not the biological location of a model role. Select a point to inspect one unit.</p></div></div><div class="explain-step"><span class="reason-num">3</span><div><h3>Follow your contribution</h3><p>Confirm or correct the word. Vocabulary and journal show how that teaching becomes part of this demo session.</p></div></div></div><p>All telemetry is synthetic. The four demo moments change the pattern, explanations and model-signal examples together.</p>`,learning:`<div class="eyebrow">The felt-state learning loop</div><h2>A small correction<br>can change the vocabulary.</h2><p>A prototype is a reference signature associated with a word you taught. The real felt-state model compares a current signature with those references.</p><div class="explain-flow"><div class="explain-step"><span class="reason-num">1</span><div><h3>Notice</h3><p>A familiar signature may match a learned word. An unfamiliar one can trigger a request for a label.</p></div></div><div class="explain-step"><span class="reason-num">2</span><div><h3>Freeze the moment</h3><p>A delayed answer must apply to the moment being asked about, not whatever the network is doing when you answer.</p></div></div><div class="explain-step"><span class="reason-num">3</span><div><h3>Teach and revisit</h3><p>A new word creates a reference; a correction can refine one. Whether recognition improves needs later observations.</p></div></div></div><div class="source-note">SOURCE: bridge/felt_state.py, server/feel.py<br>DEMO: local vocabulary and example corrections only.</div>`,architecture:`<div class="eyebrow">Default architecture / local source</div><h2>Small enough to explore.<br>Rich enough to study.</h2><p>200 sensory neurons → 500 fixed expansion units → 200 concept neurons ↔ 100 working-memory neurons.</p><p>There are 140,000 plastic weights across Expansion → Concept, Concept → Memory and Memory → Concept. The fixed Sensory → Expansion projection is additional.</p><h3>The picture is a schematic</h3><p>Positions and displayed edges are generated examples. They are not anatomical locations or the running model’s weight matrices.</p><h3>Know what a field actually means</h3><p>The current server’s “concept_membrane” field carries an accumulated activity signal, not membrane potential. Expansion is a fixed thresholded projection without a LIF membrane. A real inspector must use the right telemetry for each region.</p><div class="source-note">BASIS: Brain() defaults in brain/core.py; server/braind.py; local checkout f8ba635.</div><div class="help-links"><a href="https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html" target="_blank" rel="noreferrer">LIF neuron reference ↗</a><a href="https://github.com/Triponymous/brAIn" target="_blank" rel="noreferrer">Project source ↗</a></div>`,sensors:`<div class="eyebrow">Six kinds of input</div><h2>Patterns from a desktop.</h2><p>Keyboard activity, mouse activity, the foreground application, microphone level, idle time and time of day contribute to the model’s input.</p><div class="explain-flow"><div class="explain-step"><span class="reason-num">1</span><div><h3>Signals are not meanings</h3><p>Fast typing can accompany many different situations. A sensor reading should not become a psychological label on its own.</p></div></div><div class="explain-step"><span class="reason-num">2</span><div><h3>Keep permission and source visible</h3><p>A finished dashboard should distinguish available, disabled, missing and simulated sensors and show the sampling window for numeric readings.</p></div></div></div><div class="source-note">This prototype has no microphone, keyboard monitoring, file access or sensor connection.</div>`};
info.brain=`<div class="eyebrow">A spatial model / synthetic SNN</div><h2>A real 3D form.<br>A model you can explore.</h2><p>The fine contours are derived from three-dimensional Z-Anatomy / BodyParts3D geometry. They preserve the hemispheres, cortical folds, cerebellum and brainstem. Drag to orbit, scroll or pinch to zoom; choose Side, Front or Top for orientation.</p><h3>Shape is not connectivity</h3><p>The 1,000 selectable units and curved connections are synthetic examples of brAIn’s four computational regions. Points lie on sampled atlas surfaces, without being pushed into invented interior regions. Atlas labels follow named source structures in each hemisphere; they are orientation landmarks, not complete lobe boundaries or biological localization of SNN roles. These are not measured fiber tracts, an MRI scan, or a clinical atlas.</p><p>Point colors come directly from the same CSS palette as Circuit: blue Input, violet Expansion, mint Concepts and amber Memory. Activity changes point size and opacity, never the role color. Gold pulses still travel along illustrative sample connections. Pause or scrub the frame slider to examine them. Selecting a neuron highlights the same incoming and outgoing sample edges shown in the inspector. Circuit retains your selected unit, filter and frame.</p><h3>Controls and access</h3><p>Contours toggles the orientation drawing. Atlas labels toggles the anatomical landmarks; L and R name the source hemisphere. Connections toggles sample pathways. On the focused 3D canvas, arrow keys rotate, +/− zoom and Home resets. The Single neuron form provides precise keyboard selection.</p><div class="source-note">ANATOMY: Z-Anatomy / BodyParts3D © DBCLS, adapted under CC BY-SA 4.0.<br>MODEL DATA: synthetic; no live telemetry or anatomical mapping.</div>`;
let toastTimer;function notify(message){const t=$('.toast');t.textContent=message;t.classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>t.classList.remove('show'),3400)}
function closeDialogs(){$$('dialog[open]').forEach(d=>d.close())}
document.addEventListener('click',e=>{const b=e.target.closest('[data-action],button[data-region],[data-insight],[data-neighbor],[data-word],[data-moment],[data-journal-filter],[data-open-moment],[data-modulator],[data-event]');if(!b)return;if(b.dataset.region){setRegion(b.dataset.region);return}if(b.dataset.insight){setInsight(b.dataset.insight);return}if(b.dataset.neighbor){selectNeuron(b.dataset.neighbor);return}if(b.dataset.word){state.word=b.dataset.word;renderVocabulary();$$('[data-word]').find(el=>el.dataset.word===state.word)?.focus({preventScroll:true});return}if(b.dataset.journalFilter){state.filter=b.dataset.journalFilter;$$('[data-journal-filter]').forEach(t=>t.setAttribute('aria-pressed',t===b));renderJournal();return}if(b.dataset.event){const event=events.find(event=>event.id===b.dataset.event);state.eventId=event.id;state.moment=event.moment;renderJournal();$('[data-event="'+event.id+'"]').focus({preventScroll:true});return}if(b.dataset.moment!==undefined||b.dataset.openMoment!==undefined){state.eventId=null;state.moment=Number(b.dataset.moment??b.dataset.openMoment);renderJournal();return}if(b.dataset.modulator!==undefined){const m=modDefs[Number(b.dataset.modulator)];showInfo(`<div class="eyebrow">${m.code} / ${m.name}</div><h2>${m.title}.</h2><p>${m.detail}</p><p>The displayed value is a synthetic example on a 0–1 scale. It changes with the selected demo moment.</p><div class="source-note">MODEL BASIS: prediction-error driver in brain/core.py</div>`);return}const a=b.dataset.action;if(info[a])showInfo(info[a]);else if(a==='why')why();else if(a==='teach')teach();else if(a==='confirm')confirm();else if(a==='close')closeDialogs();else if(a==='play')play();else if(a==='export')exportSession();else if(a==='copy-link')copyView();else if(a==='desktop')desktop();else if(a==='quiet-preview'){state.quietPreview=!state.quietPreview;$('#desktop-preview-label').textContent=state.quietPreview?'Quiet workspace: preview active · click to end':'From a pattern to a helpful suggestion';closeDialogs();notify(state.quietPreview?'Preview enabled here only. Your desktop settings are unchanged.':'Demo preview ended.')}else if(a==='replay-moment'){const snap=journalSnapshot();changeMoment(snap.scenario,snap.moment);state.region=snap.region;selectNeuron(snap.selected);setRegion(snap.region,false);setFrame(snap.frame);setGraphMode(snap.graphMode);location.hash='network';syncViewURL()}else if(a?.startsWith('zoom')){if(state.graphMode==='brain'){zoomBrain(a);return;}state.zoom=a==='zoom-reset'?1:Math.max(.7,Math.min(2,state.zoom+(a==='zoom-in'?.2:-.2)));applyZoom()}});
$('#moment-select').addEventListener('change',e=>changeMoment(e.target.value));$('#frame-slider').addEventListener('input',e=>{stop();setFrame(Number(e.target.value));syncViewURL()});$('#show-edges').addEventListener('change',()=>{paintActivity();highlightSelection();syncViewURL()});$('#vocab-search').addEventListener('input',renderVocabulary);
$('#neuron-find').addEventListener('submit',e=>{e.preventDefault();const key=$('#neuron-region-select').value,r=regions.find(r=>r.key===key),index=Number($('#neuron-index').value);if(!Number.isInteger(index)||index<0||index>=r.n)return;selectNeuron(key+index)});$('#neuron-region-select').addEventListener('change',e=>{const r=regions.find(r=>r.key===e.target.value);$('#neuron-index').max=r.n-1;$('#neuron-index').value=Math.min(Number($('#neuron-index').value),r.n-1)});
$('.insight-tabs').addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const tab=e.key==='Home'?'meaning':e.key==='End'?'neuron':state.insight==='meaning'?'neuron':'meaning';setInsight(tab);$(`[data-insight="${tab}"]`).focus()});
$('#teach-form').addEventListener('submit',e=>{
  e.preventDefault();
  const label=$('#new-word').value.trim(),note=$('#word-note').value.trim();
  if(!label){$('#teach-error').hidden=false;$('#new-word').setAttribute('aria-invalid','true');$('#new-word').focus();return}
  const snap=state.frozen,existing=vocabulary.find(v=>v.label.toLowerCase()===label.toLowerCase()),previous=existing?structuredClone(existing):null;
  if(existing){existing.mods=[...snap.mods];if(note)existing.note=note;state.word=existing.label}
  else{vocabulary.push({label,note:note||'A local label for this frozen example moment.',mods:[...snap.mods],origin:'Taught in this demo',corrections:0});state.word=label}
  state.learned[slotKey(snap.scenario,snap.frame)]=state.word;
  state.confirmed.delete(slotKey(snap.scenario,snap.frame));
  recordAnnotation('You annotated “'+state.word+'”','Bound to '+snap.title.toLowerCase()+', frame '+String(snap.frame).padStart(3,'0')+'. '+(existing?'Reference revision retained in this journal.':'New local vocabulary entry.'),snap,state.word,previous);
  $('#vocab-search').value='';renderState();renderJournal();closeDialogs();
  notify('Annotation recorded for the frozen frame. Export to keep your labels and notes.');
});
$('#new-word').addEventListener('input',()=>{$('#teach-error').hidden=true;$('#new-word').removeAttribute('aria-invalid')});
$('#replay-speed').addEventListener('change',e=>{stop();state.speed=Number(e.target.value);syncViewURL()});
$('#spike-raster').addEventListener('click',e=>{const row=e.target.closest('[data-raster-unit]');if(row)selectNeuron(row.dataset.rasterUnit)});
$('.skip-link').addEventListener('click',e=>{e.preventDefault();$('#main-content').focus();$('#main-content').scrollIntoView({block:'start'})});

document.addEventListener('click',e=>{const link=e.target.closest('a[href^="#"]');if(!link||e.metaKey||e.ctrlKey||e.shiftKey||e.altKey)return;const target=link.getAttribute('href').slice(1);if(!pageCopy[target])return;e.preventDefault();if(location.hash==='#'+target)setPage();else location.hash=target});
window.addEventListener('hashchange',setPage);document.addEventListener('visibilitychange',()=>{if(document.hidden)stop()});reduced.addEventListener('change',()=>{if(reduced.matches)stop()});
new ResizeObserver(()=>{if(['overview','network'].includes(state.page)){const h=$('#graph-stage');if(!state.graph||h.clientWidth!==state.graph.w||h.clientHeight!==state.graph.h)buildGraph()}if(state.page==='network')drawRaster()}).observe($('.main'));
restoreViewURL();hydrate();
$$('[data-nav]').forEach(link=>link.setAttribute('aria-label',pageCopy[link.dataset.nav][0]));
$$('.resources button').forEach(button=>button.setAttribute('aria-label',button.textContent.trim()));
renderState();renderNeuron();renderJournal();setInsight('meaning');setPage();setFrame(state.frame);
appReady=true;renderSession();syncViewURL();
