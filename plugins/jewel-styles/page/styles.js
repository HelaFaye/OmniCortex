// Extra emotion-jewel styles for Studio's pages (the jewel-styles add-on). Loaded by viz/widgets.py while the
// add-on is on. The same drawing as styles.gd for the Godot client.
(() => {
  const ico = (() => { const g = (1+Math.sqrt(5))/2;
      let v = [[-1,g,0],[1,g,0],[-1,-g,0],[1,-g,0],[0,-1,g],[0,1,g],[0,-1,-g],[0,1,-g],[g,0,-1],[g,0,1],[-g,0,-1],[-g,0,1]];
      const f = [[0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],[1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],[3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],[4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]];
      const nrm = p => { const l = Math.hypot(...p); return p.map(x=>x/l); }; v = v.map(nrm);
      const mid = {}, nf = [];
      for(const tri of f){ const m = [[tri[0],tri[1]],[tri[1],tri[2]],[tri[2],tri[0]]].map(([a,b]) => { const k = Math.min(a,b)+'_'+Math.max(a,b);
          if(!(k in mid)){ v.push(nrm(v[a].map((x,i)=>(x+v[b][i])/2))); mid[k] = v.length-1; } return mid[k]; });
        nf.push([tri[0],m[0],m[2]],[tri[1],m[1],m[0]],[tri[2],m[2],m[1]],[m[0],m[1],m[2]]); }
      return {v, f: nf}; })();
  const spread = n => [...Array(n)].map((_,i) => { const y = 1-2*(i+0.5)/Math.max(1,n), r = Math.sqrt(Math.max(0,1-y*y)), th = Math.PI*(3-Math.sqrt(5))*i;
      return [Math.cos(th)*r, y, Math.sin(th)*r]; });
  let dirs = [];
  NSWidgets.jewelStyle('icosphere', (ctx, {cx, cy, r, cur, mix, mag, t, color, css, dark, light}) => {
      if(dirs.length !== cur.length) dirs = spread(cur.length);
      const ay = t*0.5, ax = 0.4 + 0.2*Math.sin(t*0.3), cyy = Math.cos(ay), sy = Math.sin(ay), cxx = Math.cos(ax), sx = Math.sin(ax);
      const rot = p => { const x = p[0]*cyy + p[2]*sy, z = -p[0]*sy + p[2]*cyy; return [x, p[1]*cxx - z*sx, p[1]*sx + z*cxx]; };
      const pts = [], wts = [];
      ico.v.forEach(v => { let d = 0; const w = cur.map((z,k) => { const f = Math.pow(Math.max(0, v[0]*dirs[k][0]+v[1]*dirs[k][1]+v[2]*dirs[k][2]), 3); d += z/3*f; return f*Math.abs(z); });
        const s = (0.55 + 0.35*Math.max(-0.6, Math.min(1.2, d)))*r; pts.push(rot(v.map(x=>x*s))); wts.push(w); });
      const light3 = [-0.4,-0.6,0.7].map(x=>x/Math.hypot(0.4,0.6,0.7)), faces = [];
      ico.f.forEach(tri => { const [a,b,e] = tri.map(i=>pts[i]); const u = b.map((x,i)=>x-a[i]), w = e.map((x,i)=>x-a[i]);
        let nn = [u[1]*w[2]-u[2]*w[1], u[2]*w[0]-u[0]*w[2], u[0]*w[1]-u[1]*w[0]]; const l = Math.hypot(...nn)||1; nn = nn.map(x=>x/l);
        if(nn[2] < 0) return; faces.push([(a[2]+b[2]+e[2])/3, tri, nn]); });
      faces.sort((p,q)=>p[0]-q[0]);
      for(const [_, tri, nn] of faces){ let col = [0,0,0], ws = 0;
        for(const i of tri) cur.forEach((z,k) => { let ck = color(k); if(z<0) ck = dark(ck,0.5); const w = wts[i][k]; col = col.map((v,j)=>v+ck[j]*w); ws += w; });
        col = ws < 1e-4 ? dark(mix, 0.35) : col.map(v=>v/ws); const sh = 0.35 + 0.65*Math.max(0, nn[0]*light3[0]+nn[1]*light3[1]+nn[2]*light3[2]);
        col = col.map(v=>v*sh); ctx.fillStyle = css(col, 0.95); ctx.strokeStyle = css(light(col,0.3), 0.35); ctx.lineWidth = 1;
        ctx.beginPath(); tri.forEach((i,j)=> j ? ctx.lineTo(cx+pts[i][0], cy+pts[i][1]) : ctx.moveTo(cx+pts[i][0], cy+pts[i][1])); ctx.closePath(); ctx.fill(); ctx.stroke(); }
    });
  NSWidgets.jewelStyle('tesseract', (ctx, {cx, cy, r, cur, mix, mag, t, color, css, dark, light}) => {
      let pos = 0, neg = 0; cur.forEach(z => { pos += Math.max(0,z); neg += Math.max(0,-z); });
      const axw = t*(0.3+0.5*Math.min(2,pos/3)), ayw = t*(0.2+0.5*Math.min(2,neg/3)), axy = t*0.25, proj = [];
      for(let i=0;i<16;i++){ const p = [i&1?1:-1, i&2?1:-1, i&4?1:-1, i&8?1:-1];
        const x = p[0]*Math.cos(axw) - p[3]*Math.sin(axw); let w = p[0]*Math.sin(axw) + p[3]*Math.cos(axw);
        const y = p[1]*Math.cos(ayw) - w*Math.sin(ayw); w = p[1]*Math.sin(ayw) + w*Math.cos(ayw);
        const x2 = x*Math.cos(axy) - p[2]*Math.sin(axy), z2 = x*Math.sin(axy) + p[2]*Math.cos(axy);
        const k4 = 1/(2.6-w), k3 = 1/(3 - z2*k4); proj.push([cx + x2*k4*k3*r*2.4, cy + y*k4*k3*r*2.4, w]); }
      const n = cur.length; let e = 0; ctx.lineCap = 'round';
      for(let i=0;i<16;i++) for(let b=0;b<4;b++){ const j = i ^ (1<<b); if(j < i) continue;
        const inner = proj[i][2] < 0 && proj[j][2] < 0, col = inner||!n ? mix : color(e%n), s = inner||!n ? mag : Math.min(1, Math.abs(cur[e%n])/3);
        ctx.strokeStyle = css(col, 0.35+0.65*s); ctx.lineWidth = 1+2.5*s; ctx.beginPath(); ctx.moveTo(proj[i][0],proj[i][1]); ctx.lineTo(proj[j][0],proj[j][1]); ctx.stroke(); e++; }
      ctx.fillStyle = css(light(mix,0.4), 1); proj.forEach(p => { ctx.beginPath(); ctx.arc(p[0],p[1],1.5+2*mag,0,7); ctx.fill(); });
    });
})();
