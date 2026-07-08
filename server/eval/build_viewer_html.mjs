// Emit a self-contained local HTML inspector from viewer_data.json.
// Data is inlined (works over file://, no server, no external assets).
//   node server/eval/build_viewer_html.mjs   ->  data/eval/viewer.html
import { readFileSync, writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '../..');
const D = join(ROOT, 'data/eval');
// escape '<' so any "</script>" or "<" inside article text can't break the inline <script>
const data = readFileSync(join(D, 'viewer_data.json'), 'utf-8').replace(/</g, '\\u003c');

const CSS = `
:root{--bg:#fbfbfa;--card:#fff;--ink:#1a1a1a;--mut:#6b7280;--line:#e5e7eb;--accent:#2563eb;
--pass:#15803d;--passbg:#e7f5ec;--fail:#b42318;--failbg:#fdeceb;--na:#9ca3af;--nabg:#f1f2f4;
--short:#b45309;--shortbg:#fef3e2;--teach:#7c3aed;--stud:#2563eb;--base:#6b7280;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);background:var(--bg)}
header{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid var(--line);padding:10px 16px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
header h1{font-size:15px;margin:0;font-weight:700}
.tabs{display:flex;gap:4px}
.tab{padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer;font-size:13px}
.tab.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.legend{margin-left:auto;display:flex;gap:12px;align-items:center;font-size:12px;color:var(--mut);flex-wrap:wrap}
.chip{display:inline-flex;align-items:center;gap:3px;padding:1px 6px;border-radius:4px;font-size:11px;font-weight:600;white-space:nowrap}
.chip.p{background:var(--passbg);color:var(--pass)} .chip.f{background:var(--failbg);color:var(--fail)} .chip.n{background:var(--nabg);color:var(--na)}
.badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:11px;font-weight:600}
.badge.teacher{background:#f3e8ff;color:var(--teach)} .badge.student{background:#e0edff;color:var(--stud)} .badge.baseline{background:#f1f2f4;color:var(--base)}
.badge.short{background:var(--shortbg);color:var(--short)}
main{padding:0}
/* articles view */
#articles{display:grid;grid-template-columns:320px 1fr;height:calc(100vh - 52px)}
#list{border-right:1px solid var(--line);overflow:auto;background:#fff}
#filters{position:sticky;top:0;background:#fff;padding:8px;border-bottom:1px solid var(--line);display:flex;flex-direction:column;gap:6px}
#filters input,#filters select{width:100%;padding:5px 7px;border:1px solid var(--line);border-radius:5px;font-size:13px}
.arow{padding:7px 10px;border-bottom:1px solid #f3f4f6;cursor:pointer}
.arow:hover{background:#f7f8fa}.arow.sel{background:#eef4ff;border-left:3px solid var(--accent);padding-left:7px}
.arow .t{font-weight:600;font-size:12.5px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.arow .m{font-size:11px;color:var(--mut);margin-top:2px;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
#detail{overflow:auto;padding:16px}
.arthead{margin-bottom:10px}
.arthead h2{font-size:16px;margin:0 0 4px}
.src{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0;white-space:pre-wrap;font-size:13px;max-height:260px;overflow:auto;color:#374151}
details>summary{cursor:pointer;font-weight:600;font-size:13px;color:var(--accent);margin:6px 0}
h3.sec{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);margin:18px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{border:1px solid var(--line);padding:4px 7px;text-align:left;vertical-align:top}
th{background:#f7f8fa;font-weight:600;position:sticky;top:0}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.cls-match{background:var(--passbg)} .cls-diff{background:#fff7e6}
.gen{background:#fff;border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:8px 0}
.gen .gh{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.gen .gh .nm{font-weight:700}
.gen .gh .meta{margin-left:auto;font-size:11px;color:var(--mut)}
.gen .sm{font-size:13.5px;margin:4px 0}
.gen .tp{margin:4px 0}
.gen .strip{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.topic{display:inline-block;background:#eef2f7;color:#475569;border-radius:10px;padding:1px 8px;font-size:11px;margin:1px}
.pass-mini{font-weight:700}
/* scorecard + analytics */
.wrap{max-width:1100px;margin:0 auto;padding:18px}
.card{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px 16px;margin:14px 0}
.card h3{margin:0 0 10px;font-size:14px}
.bar{height:14px;background:var(--accent);border-radius:3px;display:inline-block;vertical-align:middle}
.note{font-size:12px;color:var(--mut)}
.hide{display:none!important}
.kpi{display:inline-block;font-weight:700}
.wide{overflow-x:auto}
`;

const JS = `
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const ARMS=DATA.meta.arms, ARMKEY=Object.fromEntries(ARMS.map(a=>[a.key,a])), CHECKS=DATA.meta.checks, FIELDS=DATA.meta.catFields;
const CK=CHECKS.map(c=>c[0]);
function chip(v){const c=v===1?'p':v===0?'f':'n';const t=v===1?'✓':v===0?'✗':'–';return '<span class="chip '+c+'">'+t+'</span>';}
function esc(s){return (s==null?'':String(s)).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]));}

// ---------- per-arm aggregates computed from per-item grades ----------
function aggregates(){
  const out={};
  for(const a of ARMS){const key=a.key;let items=0,passSum=0;const per={};for(const c of CK)per[c]={p:0,n:0};let sv=0,svn=0;const lats=[];
    for(const art of DATA.articles){const g=art.checklist[key];const gen=art.arms[key];if(gen){svn++;if(gen.schemaValid)sv++;if(gen.durationMs)lats.push(gen.durationMs);}
      if(!g)continue;items++;let passed=0,tot=0;
      for(const c of CK){if(c==='topics_cover')continue;tot++;if(g[c]===1)passed++;}
      passSum+=passed/tot;
      for(const c of CK){if(g[c]==null)continue;per[c].n++;if(g[c]===1)per[c].p++;}}
    lats.sort((x,y)=>x-y);const lat=lats.length?lats[Math.floor(lats.length/2)]:null;
    out[key]={checklist:items?100*passSum/items:null,per,schema:svn?100*sv/svn:null,items,lat};}
  return out;
}

// ---------- ARTICLES VIEW ----------
let sel=0;
function artMeta(a){const short=a.chars<=1200;return '<span>#'+a.id+'</span> · <span>'+esc(a.feed)+'</span> · <span>'+a.chars+'c</span>'+(short?' <span class="badge short">SHORT</span>':'');}
function renderList(){
  const q=($('#q').value||'').toLowerCase(),ff=$('#feed').value,lf=$('#len').value;
  const rows=DATA.articles.map((a,i)=>({a,i})).filter(({a})=>{
    if(ff&&a.feed!==ff)return false;
    if(lf==='short'&&a.chars>1200)return false; if(lf==='long'&&a.chars<=1200)return false;
    if(q&&!(a.title.toLowerCase().includes(q)||String(a.id).includes(q)||a.feed.toLowerCase().includes(q)))return false;
    return true;});
  $('#list-body').innerHTML=rows.map(({a,i})=>'<div class="arow'+(i===sel?' sel':'')+'" data-i="'+i+'"><div class="t">'+esc(a.title)+'</div><div class="m">'+artMeta(a)+'</div></div>').join('');
  $$('#list-body .arow').forEach(el=>el.onclick=()=>{sel=+el.dataset.i;renderList();renderDetail();});
  $('#count').textContent=rows.length+' / '+DATA.articles.length;
}
function clsTable(a){
  let h='<div class="wide"><table><thead><tr><th>Arm</th>'+FIELDS.map(f=>'<th>'+f+'</th>').join('')+'</tr></thead><tbody>';
  h+='<tr><td><b>teacher labels (ref)</b></td>'+FIELDS.map(f=>'<td><b>'+esc(a.teacherLabels[f])+'</b></td>').join('')+'</tr>';
  for(const arm of ARMS){const lab=a.classLabels[arm.key];if(!lab)continue;
    h+='<tr><td>'+esc(arm.label)+' <span class="badge '+arm.kind+'">'+arm.kind+'</span></td>'+
      FIELDS.map(f=>{const v=lab[f],ref=a.teacherLabels[f];const cls=v==null?'':(String(v).toLowerCase()===String(ref).toLowerCase()?'cls-match':'cls-diff');return '<td class="'+cls+'">'+esc(v)+'</td>';}).join('')+'</tr>';}
  return h+'</tbody></table></div>';
}
function genCard(a,arm){const g=a.arms[arm.key];if(!g)return '';const ck=a.checklist[arm.key]||{};
  const strip=CK.map(c=>'<span title="'+esc((CHECKS.find(x=>x[0]===c)||[])[1])+'">'+chip(ck[c])+'<span style="font-size:10px;color:var(--mut)"> '+c+'</span></span>').join(' ');
  const passed=CK.filter(c=>c!=='topics_cover').filter(c=>ck[c]===1).length;
  return '<div class="gen"><div class="gh"><span class="nm">'+esc(arm.label)+'</span><span class="badge '+arm.kind+'">'+arm.kind+'</span>'+
    '<span class="note">'+esc(arm.note)+'</span><span class="meta">'+(g.schemaValid?'schema ✓':'schema ✗')+
    (g.durationMs?' · '+g.durationMs+'ms':'')+' · checklist '+passed+'/8</span></div>'+
    '<div class="sm">'+esc(g.summary||'(no summary parsed)')+'</div>'+
    '<div class="tp">'+(g.topics||[]).map(t=>'<span class="topic">'+esc(t)+'</span>').join('')+'</div>'+
    '<div class="strip">'+strip+'</div></div>';
}
function renderDetail(){const a=DATA.articles[sel];if(!a){$('#detail').innerHTML='';return;}
  const order=['teacher','teacher_llama','distil','tuned_rss_tuned_s1','tuned_rss_tuned_s2','tuned_rss_tuned_s3','tuned_rss_llama_s1','tuned_rss_llama_s2','tuned_rss_llama_s3','base','fewshot','constrained'];
  const arms=order.map(k=>ARMKEY[k]).filter(Boolean);
  $('#detail').innerHTML='<div class="arthead"><h2>'+esc(a.title)+'</h2><div class="note">#'+a.id+' · '+esc(a.feed)+' · '+a.chars+' chars'+(a.chars<=1200?' · <span class="badge short">SHORT / thin source</span>':'')+'</div></div>'+
    '<details><summary>Show source article</summary><div class="src">'+esc(a.input)+'</div></details>'+
    '<h3 class="sec">Classification — predicted labels (green = matches teacher, amber = differs)</h3>'+clsTable(a)+
    '<h3 class="sec">Generations &amp; summary checklist (3-judge panel majority)</h3>'+arms.map(arm=>genCard(a,arm)).join('');
}

// ---------- SCORECARD VIEW ----------
function bar(v,max=100,w=90){return '<span class="bar" style="width:'+(v/max*w)+'px"></span>';}
function renderScore(){const ag=aggregates();const sc=DATA.analytics.scores;
  const order=['teacher','teacher_llama','distil','tuned_rss_tuned_s1','tuned_rss_tuned_s2','tuned_rss_tuned_s3','tuned_rss_llama_s1','tuned_rss_llama_s2','tuned_rss_llama_s3','base','fewshot','constrained'];
  const sl=sc.structure_length||{},cls=sc.classification?.perArm||{};
  let t1='<div class="wide"><table><thead><tr><th>Arm</th><th class="num">Checklist %</th><th></th><th class="num">Faithful %</th><th class="num">Topics %</th><th class="num">Class macro %</th><th class="num">Schema %</th><th class="num">p50 ms</th></tr></thead><tbody>';
  for(const k of order){const a=ARMKEY[k];if(!a)continue;const g=ag[k];const fa=g.per.faithful.n?100*g.per.faithful.p/g.per.faithful.n:null;
    const tc=g.per.topics_cover.n?100*g.per.topics_cover.p/g.per.topics_cover.n:null;
    t1+='<tr><td>'+esc(a.label)+' <span class="badge '+a.kind+'">'+a.kind+'</span></td>'+
      '<td class="num">'+fmt(g.checklist)+'</td><td>'+bar(g.checklist||0)+'</td>'+
      '<td class="num">'+fmt(fa)+'</td><td class="num">'+fmt(tc)+'</td>'+
      '<td class="num">'+fmt(cls[k]?.macroAcc)+'</td><td class="num">'+fmt(sl[k]?.schemaValidPct)+'</td>'+
      '<td class="num">'+(g.lat!=null?g.lat:'–')+'</td></tr>';}
  t1+='</tbody></table></div>';
  // per-check
  const pc=sc.checklist?.__perCheck||{};
  let t2='<div class="wide"><table><thead><tr><th>Check</th><th class="num">Pass %</th><th></th></tr></thead><tbody>'+
    CK.map(c=>'<tr><td>'+c+'</td><td class="num">'+fmt(pc[c])+'</td><td>'+bar(pc[c]||0)+'</td></tr>').join('')+'</tbody></table></div>';
  // classification byField
  const bf=Object.entries(cls);
  let t3='<div class="wide"><table><thead><tr><th>Arm</th>'+FIELDS.map(f=>'<th class="num">'+f+'</th>').join('')+'<th class="num">macro</th></tr></thead><tbody>';
  for(const k of order){const a=ARMKEY[k];if(!a||!cls[k])continue;t3+='<tr><td>'+esc(a.label)+'</td>'+FIELDS.map(f=>'<td class="num">'+fmt(cls[k].byField?.[f])+'</td>').join('')+'<td class="num"><b>'+fmt(cls[k].macroAcc)+'</b></td></tr>';}
  t3+='</tbody></table></div>';
  const agr=sc.classification?.agreement||{};
  let t4='<table><thead><tr><th>Field</th>'+FIELDS.map(f=>'<th class="num">'+f+'</th>').join('')+'</tr></thead><tbody><tr><td>judge agreement</td>'+FIELDS.map(f=>'<td class="num">'+(agr[f]?.rawAgreement??'–')+'</td>').join('')+'</tr></tbody></table>';
  $('#score').innerHTML='<div class="wrap">'+
    '<div class="card"><h3>Per-arm summary &amp; classification</h3><p class="note">Checklist % and Faithful % recomputed live from per-item panel-majority grades; macro/schema from scores.json.</p>'+t1+'</div>'+
    '<div class="card"><h3>Per-check pass-rate (all arms pooled)</h3>'+t2+'</div>'+
    '<div class="card"><h3>Per-field classification accuracy vs judge proxy-gold</h3>'+t3+'<div style="margin-top:10px">'+t4+'</div></div>';
}
function fmt(v){return v==null||isNaN(v)?'–':(+v).toFixed(1);}

// ---------- ANALYTICS VIEW ----------
function renderAnalytics(){const A=DATA.analytics,sc=A.scores,det=A.detail,rt=A.router;
  let judges='<div class="card"><h3>Judge panel</h3><table><thead><tr><th>id</th><th>provider</th><th>model</th></tr></thead><tbody>'+
    DATA.meta.judges.map(j=>'<tr><td>'+j.id+'</td><td>'+j.provider+'</td><td>'+esc(j.model)+'</td></tr>').join('')+'</tbody></table>'+
    '<p class="note">'+esc(DATA.meta.builtAtNote)+'</p></div>';
  const nc=sc.negative_control;
  let neg=nc?'<div class="card"><h3>Negative control</h3><p>Faithful on <b>mismatched</b> article: <span class="kpi">'+nc.faithfulPct_on_mismatch+'%</span> (n='+nc.n+'). '+esc(nc.note)+'</p></div>':'';
  let seed='';if(det.seedAgreement){seed='<div class="card"><h3>Seed unanimity predicts label correctness</h3><table><thead><tr><th>field</th><th class="num">unanimous n</th><th class="num">acc</th><th class="num">split n</th><th class="num">acc</th></tr></thead><tbody>'+
    Object.entries(det.seedAgreement).map(([f,v])=>'<tr><td>'+f+'</td><td class="num">'+v.unanimousN+'</td><td class="num">'+v.unanimousAcc+'</td><td class="num">'+v.splitN+'</td><td class="num">'+v.splitAcc+'</td></tr>').join('')+'</tbody></table></div>';}
  let dis='';if(det.judgeDisagreementByCheck){dis='<div class="card"><h3>Judge disagreement by check (% items judges split)</h3><table><tbody>'+
    Object.entries(det.judgeDisagreementByCheck).map(([c,v])=>'<tr><td>'+c+'</td><td class="num">'+v+'%</td></tr>').join('')+'</tbody></table></div>';}
  let router='';if(rt){const R=rt.routers||{};
    router='<div class="card"><h3>Routing configurations (composite, post-hoc)</h3><div class="wide"><table><thead><tr><th>config</th><th>short arm</th><th class="num">checklist</th><th class="num">Δ vs tuned</th><th class="num">faithful</th><th class="num">batch/500 min</th></tr></thead><tbody>'+
      Object.entries(R).map(([n,v])=>'<tr><td>'+n+'</td><td>'+v.shortArm+'</td><td class="num">'+(v.checklist?.mean??'–')+'</td><td class="num">'+(v.pairedVsTuned?.delta??'–')+'</td><td class="num">'+v.faithful+'</td><td class="num">'+v.batch500min+'</td></tr>').join('')+
      '</tbody></table></div><p class="note">B\\'(short→Llama-8B teacher) is computed separately (server/eval/router_bprime.mjs): checklist 71.6, faithful 82.8, ~14 min.</p></div>';}
  let urg='';if(det.urgency){const u=det.urgency;urg='<div class="card"><h3>Urgency: gold distribution &amp; majority baseline</h3><p class="note">gold '+JSON.stringify(u.goldDist)+' · majority baseline '+u.majorityBaseline+'%</p></div>';}
  $('#analytics').innerHTML='<div class="wrap">'+judges+neg+seed+urg+dis+router+'</div>';
}

// ---------- tabs ----------
function show(v){$$('.tab').forEach(t=>t.classList.toggle('active',t.dataset.v===v));
  ['articles','score','analytics'].forEach(id=>$('#'+id).classList.toggle('hide',id!==v));
  if(v==='score')renderScore(); if(v==='analytics')renderAnalytics();}
$$('.tab').forEach(t=>t.onclick=()=>show(t.dataset.v));
// feed filter options
const feeds=[...new Set(DATA.articles.map(a=>a.feed))].sort();
$('#feed').innerHTML='<option value="">all feeds ('+feeds.length+')</option>'+feeds.map(f=>'<option>'+esc(f)+'</option>').join('');
['q','feed','len'].forEach(id=>$('#'+id).oninput=renderList);
renderList();renderDetail();show('articles');
`;

const html = `<!doctype html><html><head><meta charset="utf-8"><title>Distillation eval — local inspector</title>
<style>${CSS}</style></head><body>
<header>
  <h1>Distillation eval · local inspector</h1>
  <div class="tabs">
    <div class="tab active" data-v="articles">Articles</div>
    <div class="tab" data-v="score">Scorecard</div>
    <div class="tab" data-v="analytics">Analytics</div>
  </div>
  <div class="legend"><span class="chip p">✓</span>pass <span class="chip f">✗</span>fail <span class="chip n">–</span>n/a
    <span class="badge teacher">teacher</span><span class="badge student">student</span><span class="badge baseline">baseline</span></div>
</header>
<main>
  <div id="articles">
    <div id="list">
      <div id="filters">
        <input id="q" placeholder="search title / id / feed">
        <select id="feed"></select>
        <select id="len"><option value="">all lengths</option><option value="short">short ≤1200</option><option value="long">long &gt;1200</option></select>
        <div class="note"><span id="count"></span> articles</div>
      </div>
      <div id="list-body"></div>
    </div>
    <div id="detail"></div>
  </div>
  <div id="score" class="hide"></div>
  <div id="analytics" class="hide"></div>
</main>
<script>const DATA=${data};</script>
<script>${JS}</script>
</body></html>`;

writeFileSync(join(D, 'viewer.html'), html);
console.log(`-> data/eval/viewer.html (${(html.length / 1e6).toFixed(2)} MB)`);
