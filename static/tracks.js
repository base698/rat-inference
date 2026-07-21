(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const canvas = $('replayCanvas');
  const ctx = canvas.getContext('2d');
  const colors = ['#50b5ff','#ff7f66','#57d68d','#d58cff','#ffd166','#6ee7e0','#ff8bc2','#a3e635'];
  let recordingId = null, replay = null, frames = [], index = 0, mode = '2d';
  let trackIds = [];
  let playing = false, speed = 1, animationStart = 0, playbackStartIndex = 0;

  const colorFor = id => colors[(Number(id || 1) - 1) % colors.length];
  const frameTime = frame => Number(frame?.monotonic_time || 0);
  const firstTime = () => frames.length ? frameTime(frames[0]) : 0;
  const lastTime = () => frames.length ? frameTime(frames[frames.length - 1]) : 0;
  const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
  const selectedTrack = () => $('trackSelect').value;
  const visibleId = id => selectedTrack() === 'all' || Number(selectedTrack()) === Number(id);
  const replayFps = () => {
    const duration = lastTime() - firstTime();
    return Number.isFinite(duration) && duration > 0
      ? clamp((frames.length - 1) / duration, 1, 60)
      : 10;
  };
  const positiveNumber = value => {
    const number = Number(value);
    return Number.isFinite(number) && number > 0 ? number : null;
  };
  function recordingImageSize(frame = null) {
    const metadata = replay?.metadata || {};
    const parameters = metadata.parameters || {};
    const directSize = metadata.image_size || parameters.image_size || frame?.image_size || {};
    const width = positiveNumber(
      directSize.width ?? metadata.image_width ?? parameters.image_width
    );
    const height = positiveNumber(
      directSize.height ?? metadata.image_height ?? parameters.image_height
    );
    if (width && height) return {width, height};
    return inferImageSize();
  }
  function inferImageSize() {
    let maxX = 0, maxY = 0;
    frames.forEach(frame => {
      (frame.measurements || []).forEach(measurement => {
        const bbox = measurement.bbox;
        if (Array.isArray(bbox) && bbox.length === 4) {
          maxX = Math.max(maxX, Number(bbox[0]) || 0, Number(bbox[2]) || 0);
          maxY = Math.max(maxY, Number(bbox[1]) || 0, Number(bbox[3]) || 0);
        }
        const center = measurement.center;
        if (Array.isArray(center) && center.length >= 2) {
          maxX = Math.max(maxX, Number(center[0]) || 0);
          maxY = Math.max(maxY, Number(center[1]) || 0);
        }
      });
    });
    const sizes = [[640,480], [960,720], [1280,960], [1920,1080]];
    const match = sizes.find(([width, height]) => maxX <= width && maxY <= height);
    return match ? {width: match[0], height: match[1]} : {width: 640, height: 480};
  }

  async function jsonFetch(url, options) {
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || `${response.status} ${response.statusText}`);
    return data;
  }

  async function loadCatalog() {
    try {
      const data = await jsonFetch('/api/track-recordings');
      const select = $('recordingSelect');
      select.innerHTML = '';
      if (!data.recordings.length) {
        select.innerHTML = '<option value="">No recordings yet</option>';
        recordingId = null; replay = null; frames = []; index = 0;
        rebuildTrackOptions();
        $('timeline').max = 0;
        $('timeline').value = 0;
        $('duration').textContent = '0.00s';
        $('deleteRecordingButton').disabled = true;
        $('status').textContent = 'Start a recording from the main control page.';
        drawEmpty('No saved track recordings');
        updateMetrics(null);
        return;
      }
      data.recordings.forEach(item => {
        const option = document.createElement('option');
        option.value = item.id;
        const date = item.started_at ? new Date(item.started_at).toLocaleString() : item.id;
        option.textContent = `${date} · ${item.frame_count || 0} frames`;
        select.appendChild(option);
      });
      await loadRecording(select.value);
    } catch (error) {
      $('status').textContent = error.message;
      $('deleteRecordingButton').disabled = true;
      drawEmpty('Could not load recording catalog');
    }
  }

  async function loadRecording(id) {
    if (!id) return;
    pause();
    $('status').textContent = 'Loading recording…';
    try {
      replay = await jsonFetch(`/api/track-recordings/${encodeURIComponent(id)}`);
      recordingId = id;
      frames = replay.frames || [];
      index = 0;
      setParameterInputs(replay.metadata?.parameters || {});
      rebuildTrackOptions();
      $('timeline').max = Math.max(0, frames.length - 1);
      $('timeline').value = 0;
      $('duration').textContent = formatSeconds(lastTime() - firstTime());
      $('deleteRecordingButton').disabled = false;
      $('status').textContent = `${frames.length} observation frames loaded.`;
      render();
    } catch (error) {
      $('status').textContent = error.message;
      recordingId = null;
      frames = [];
      $('deleteRecordingButton').disabled = true;
      render();
    }
  }

  function setParameterInputs(p) {
    $('confirmHits').value = p.confirm_hits ?? 3;
    $('gateDistance').value = p.gate_distance_mm ?? 750;
    $('maxMisses').value = p.max_misses ?? 5;
    $('deleteAfter').value = p.delete_after_seconds ?? 1.5;
    $('reidentifyAfter').value = p.reidentify_after_seconds ?? 8;
    $('processNoise').value = p.process_acceleration_std_mm_s2 ?? 300;
    $('confidenceDecay').value = p.confidence_decay ?? 0.85;
  }

  function parameters() {
    return {
      confirm_hits: Number($('confirmHits').value),
      gate_distance_mm: Number($('gateDistance').value),
      max_misses: Number($('maxMisses').value),
      delete_after_seconds: Number($('deleteAfter').value),
      reidentify_after_seconds: Number($('reidentifyAfter').value),
      process_acceleration_std_mm_s2: Number($('processNoise').value),
      confidence_decay: Number($('confidenceDecay').value),
    };
  }

  async function reprocess() {
    if (!recordingId) return;
    pause();
    $('reprocessButton').disabled = true;
    $('status').textContent = 'Reprocessing saved observations…';
    try {
      replay = await jsonFetch(`/api/track-recordings/${encodeURIComponent(recordingId)}/reprocess`, {
        method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(parameters())
      });
      frames = replay.frames || [];
      index = Math.min(index, Math.max(0, frames.length - 1));
      rebuildTrackOptions();
      $('status').textContent = `Reprocessed ${frames.length} frames. No robot commands were sent.`;
      render();
    } catch (error) { $('status').textContent = error.message; }
    finally { $('reprocessButton').disabled = false; }
  }

  async function deleteRecording() {
    const id = $('recordingSelect').value || recordingId;
    if (!id) return;
    pause();
    if (!window.confirm(`Delete recording ${id}? This cannot be undone.`)) return;
    const button = $('deleteRecordingButton');
    button.disabled = true;
    $('status').textContent = `Deleting ${id}…`;
    try {
      await jsonFetch(`/api/track-recordings/${encodeURIComponent(id)}`, {method: 'DELETE'});
      if (recordingId === id) {
        recordingId = null; replay = null; frames = []; index = 0;
      }
      await loadCatalog();
      $('status').textContent = `Deleted recording ${id}.`;
    } catch (error) {
      $('status').textContent = error.message;
      button.disabled = !$('recordingSelect').value;
    }
  }

  function rebuildTrackOptions() {
    const prior = selectedTrack();
    const idSet=new Set();
    frames.forEach(frame=>(frame.tracks||[]).forEach(track=>idSet.add(Number(track.id))));
    const ids=[...idSet].sort((a,b)=>a-b);
    trackIds=ids;
    $('trackSelect').innerHTML = '<option value="all">All tracks</option>' + ids.map(id => `<option value="${id}">Track ${id}</option>`).join('');
    if (prior === 'all' || ids.includes(Number(prior))) $('trackSelect').value = prior || 'all';
    $('legend').innerHTML = ids.map(id => `<span style="--c:${colorFor(id)}">Track ${id}</span>`).join('');
  }

  function play() {
    if (!frames.length) return;
    if (playing) return;
    if (index >= frames.length - 1) {
      index = 0;
      $('timeline').value = index;
      render();
    }
    playing = true;
    resetPlaybackClock();
    $('playButton').classList.add('active');
    $('pauseButton').classList.remove('active');
    requestAnimationFrame(tick);
  }
  function resetPlaybackClock() {
    animationStart = performance.now();
    playbackStartIndex = index;
  }
  function pause() {
    playing = false;
    $('playButton').classList.remove('active');
    $('pauseButton').classList.add('active');
  }
  function tick(now) {
    if (!playing) return;
    const elapsed = Math.max(0, (now - animationStart) / 1000);
    const nextIndex = Math.min(
      frames.length - 1,
      playbackStartIndex + Math.floor(elapsed * replayFps() * speed)
    );
    index = nextIndex;
    if (index >= frames.length - 1) pause();
    $('timeline').value = index;
    render();
    if (playing) requestAnimationFrame(tick);
  }

  function resizeCanvas() {
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    const width = Math.round(rect.width * ratio), height = Math.round(rect.height * ratio);
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    ctx.setTransform(ratio,0,0,ratio,0,0);
    return {w:rect.width,h:rect.height};
  }
  function clear(w,h) { ctx.fillStyle='#05070a'; ctx.fillRect(0,0,w,h); }
  function drawEmpty(text) { const {w,h}=resizeCanvas(); clear(w,h); ctx.fillStyle='#77869a'; ctx.textAlign='center'; ctx.font='15px system-ui'; ctx.fillText(text,w/2,h/2); }

  function render() {
    if (!frames.length) { drawEmpty('No observation frames'); updateMetrics(null); return; }
    const frame = frames[index];
    mode === '2d' ? draw2d(frame) : draw3d(frame);
    updateMetrics(frame);
  }

  function assignmentMap(frame) {
    const map = new Map();
    (frame.assignments || []).forEach(a => map.set(Number(a.detection_index), Number(a.track_id)));
    return map;
  }
  function measurementVisible(map, detectionIndex) {
    const selected = selectedTrack();
    return selected === 'all' || map.get(detectionIndex) === Number(selected);
  }

  function draw2d(frame) {
    const {w,h}=resizeCanvas(); clear(w,h);
    const imageSize=recordingImageSize(frame), imageW=imageSize.width, imageH=imageSize.height, sx=w/imageW, sy=h/imageH;
    const heatCellPx=20, cols=Math.ceil(imageW/heatCellPx), rows=Math.ceil(imageH/heatCellPx);
    const heat=new Float32Array(cols*rows);
    let peak=1;
    const frameStride=Math.max(1,Math.floor((index+1)/2000));
    for (let fi=0; fi<=index; fi+=frameStride) {
      const f=frames[fi], map=assignmentMap(f);
      (f.measurements || []).forEach((m,di) => {
        const id=map.get(di); if (!measurementVisible(map,di)) return;
        addHeatForMeasurement(m);
      });
    }
    const cw=w/cols,ch=h/rows;
    heat.forEach((value,i)=>{ if(!value)return; const alpha=.08+.58*Math.sqrt(value/peak); ctx.fillStyle=`rgba(255,84,62,${alpha})`; ctx.fillRect((i%cols)*cw,Math.floor(i/cols)*ch,cw+1,ch+1); });
    ctx.strokeStyle='rgba(255,255,255,.12)'; ctx.strokeRect(.5,.5,w-1,h-1);
    const map=assignmentMap(frame);
    (frame.measurements || []).forEach((m,di)=>{
      const id=map.get(di); if (!measurementVisible(map,di)) return;
      if (!m.bbox) return;
      const [x1,y1,x2,y2]=m.bbox, color=colorFor(id || di+1);
      ctx.strokeStyle=color; ctx.lineWidth=2; ctx.strokeRect(x1*sx,y1*sy,(x2-x1)*sx,(y2-y1)*sy);
      ctx.fillStyle=color; ctx.font='bold 13px ui-monospace'; ctx.fillText(id ? `T${id}` : 'unmatched',x1*sx+3,Math.max(14,y1*sy-4));
    });

    function addHeatForMeasurement(m) {
      const box = normalizedBox(m.bbox);
      if (box) {
        const [left, top, right, bottom] = box;
        const minX = clamp(Math.floor(left / imageW * cols), 0, cols - 1);
        const maxX = clamp(Math.floor((right - 1) / imageW * cols), 0, cols - 1);
        const minY = clamp(Math.floor(top / imageH * rows), 0, rows - 1);
        const maxY = clamp(Math.floor((bottom - 1) / imageH * rows), 0, rows - 1);
        for (let y=minY; y<=maxY; y++) {
          for (let x=minX; x<=maxX; x++) addHeat(x, y);
        }
        return;
      }
      const c=m.center; if (!c) return;
      const x=clamp(Math.floor(Number(c[0])/imageW*cols),0,cols-1);
      const y=clamp(Math.floor(Number(c[1])/imageH*rows),0,rows-1);
      addHeat(x, y);
    }
    function normalizedBox(bbox) {
      if (!Array.isArray(bbox) || bbox.length !== 4) return null;
      const values=bbox.map(Number);
      if (!values.every(Number.isFinite)) return null;
      const left=clamp(Math.min(values[0],values[2]),0,imageW);
      const right=clamp(Math.max(values[0],values[2]),0,imageW);
      const top=clamp(Math.min(values[1],values[3]),0,imageH);
      const bottom=clamp(Math.max(values[1],values[3]),0,imageH);
      return right > left && bottom > top ? [left,top,right,bottom] : null;
    }
    function addHeat(x, y) {
      const offset=y*cols+x;
      heat[offset] += 1;
      peak=Math.max(peak,heat[offset]);
    }
  }

  function allPositions() {
    const stride=Math.max(1,Math.floor(frames.length/2000));
    const positions=[];
    for(let fi=0;fi<frames.length;fi+=stride) {
      (frames[fi].tracks||[]).filter(t=>visibleId(t.id)&&validPoint(t.position_mm)).forEach(t=>positions.push(t.position_mm));
    }
    return positions;
  }
  function validPoint(point) {
    return Array.isArray(point) && point.length === 3 && point.every(value => Number.isFinite(Number(value)));
  }
  function draw3d(frame) {
    const {w,h}=resizeCanvas(); clear(w,h);
    const positions=allPositions();
    const extent=Math.max(500,...positions.flatMap(p=>p.map(v=>Math.abs(Number(v)))));
    const scale=Math.min(w,h)/(extent*2.7);
    const project=p=>[w*.52+(p[1]-p[0]*.42)*scale,h*.72-(p[2]+p[0]*.32)*scale];
    ctx.lineWidth=1; ctx.strokeStyle='rgba(130,150,175,.18)';
    for(let n=-4;n<=4;n++){ const v=extent*n/4; let a=project([-extent,v,0]),b=project([extent,v,0]); line(a,b); a=project([v,-extent,0]);b=project([v,extent,0]);line(a,b); }
    [['x forward',[extent,0,0],'#ff7f66'],['y left',[0,extent,0],'#57d68d'],['z up',[0,0,extent],'#50b5ff']].forEach(([,p,c])=>{ctx.strokeStyle=c;line(project([0,0,0]),project(p));});
    [['x forward','#ff7f66'],['y left','#57d68d'],['z up','#50b5ff']].forEach(([label,c],i)=>{ctx.fillStyle=c;ctx.fillText(label,w-82,24+i*18);});
    const ids=trackIds;
    const trailStride=Math.max(1,Math.floor((index+1)/2000));
    const visibleIds=new Set(ids.filter(visibleId));
    const trails=new Map([...visibleIds].map(id=>[id,[]]));
    for(let fi=0;fi<=index;fi+=trailStride){
      for(const track of (frames[fi].tracks||[])){
        const id=Number(track.id);
        if(visibleIds.has(id) && validPoint(track.position_mm)) trails.get(id).push(project(track.position_mm));
      }
    }
    trails.forEach((points,id)=>{
      ctx.strokeStyle=colorFor(id);ctx.lineWidth=2;ctx.beginPath();
      points.forEach((point,i)=>i?ctx.lineTo(point[0],point[1]):ctx.moveTo(point[0],point[1]));
      ctx.stroke();
    });
    const map=assignmentMap(frame);
    (frame.measurements||[]).forEach((m,di)=>{if(!measurementVisible(map,di)||!validPoint(m.base_point_mm))return;const q=project(m.base_point_mm);ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(q[0]-4,q[1]-4);ctx.lineTo(q[0]+4,q[1]+4);ctx.moveTo(q[0]+4,q[1]-4);ctx.lineTo(q[0]-4,q[1]+4);ctx.stroke();});
    (frame.tracks||[]).filter(t=>visibleId(t.id)&&validPoint(t.position_mm)).forEach(t=>{
      const q=project(t.position_mm), color=colorFor(t.id), odd=Number(t.id)%2===1;
      ctx.fillStyle=color; ctx.beginPath(); ctx.arc(q[0],q[1],t.status==='confirmed'?7:5,0,Math.PI*2); ctx.fill();
      const label=`T${t.id} ${t.status}`; ctx.font='bold 12px ui-monospace';
      const labelX=odd?q[0]+10:q[0]-ctx.measureText(label).width-10, labelY=q[1]+(odd?-14:26);
      ctx.fillStyle='rgba(5,7,10,.82)'; ctx.fillRect(labelX-3,labelY-12,ctx.measureText(label).width+6,16);
      ctx.fillStyle=color; ctx.fillText(label,labelX,labelY);
      if(validPoint(t.velocity_mm_s)){const end=project(t.position_mm.map((v,i)=>Number(v)+Number(t.velocity_mm_s[i])*.4));ctx.strokeStyle=color;ctx.lineWidth=2;line(q,end);}
    });
    function line(a,b){ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}
  }

  function updateMetrics(frame) {
    const elapsed=frames.length ? frameTime(frames[index])-firstTime() : 0;
    $('elapsed').textContent=formatSeconds(elapsed);
    $('frameMetric').textContent=`${frames.length ? index+1 : 0} / ${frames.length}`;
    $('trackMetric').textContent=frame ? (frame.tracks||[]).filter(t=>visibleId(t.id)).length : 0;
    const visibleMeasurements=frame
      ? (frame.measurements||[]).filter((_,di)=>measurementVisible(assignmentMap(frame),di)).length
      : 0;
    $('measurementMetric').textContent=visibleMeasurements;
    $('selectedMetric').textContent=frame?.selected_track_id ?? '—';
    const imageSize=frame ? recordingImageSize(frame) : null;
    $('frameBadge').textContent=frame ? `${mode.toUpperCase()} · ${imageSize.width}x${imageSize.height} · ${frame.recorded_at || ''}` : 'No recording loaded';
  }
  const formatSeconds=value=>`${Math.max(0,Number(value)||0).toFixed(2)}s`;

  $('recordingSelect').addEventListener('change',e=>loadRecording(e.target.value));
  $('trackSelect').addEventListener('change',render);
  $('speedSelect').addEventListener('change',e=>{speed=Number(e.target.value);if(playing)resetPlaybackClock();});
  $('playButton').addEventListener('click',play); $('pauseButton').addEventListener('click',pause);
  $('timeline').addEventListener('input',e=>{pause();index=Number(e.target.value);render();});
  $('mode2d').addEventListener('click',()=>{mode='2d';$('mode2d').classList.add('active');$('mode3d').classList.remove('active');render();});
  $('mode3d').addEventListener('click',()=>{mode='3d';$('mode3d').classList.add('active');$('mode2d').classList.remove('active');render();});
  $('reprocessButton').addEventListener('click',reprocess);
  $('deleteRecordingButton').addEventListener('click',deleteRecording);
  window.addEventListener('resize',render);
  loadCatalog();
})();
