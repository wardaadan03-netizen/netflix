<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Netflix Content Strategy: Data Science Capstone</title>
<style>
:root{box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px);--bg:#fff;--fg:#1b1b1b;--mut:#666;--card:#f4f4f5;--ln:#ddd;--ink:#221f1f}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141414;--fg:#eee;--mut:#aaa;--card:#222;--ln:#383838;--ink:#ddd}}
:root[data-theme=dark]{--bg:#141414;--fg:#eee;--mut:#aaa;--card:#222;--ln:#383838;--ink:#ddd}
html{scroll-padding-top:env(safe-area-inset-top,0px)}*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:26px;margin:0}h2{font-size:19px;margin:22px 0 6px}h3{font-size:16px;margin:18px 0 6px}
.m{color:var(--mut);margin:4px 0 14px}
.k{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:16px}
.k div{background:var(--card);border-radius:10px;padding:10px 12px}.k b{display:block;font-size:22px;color:#e50914}.k span{font-size:12px;color:var(--mut)}
.seg{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
.seg button{border:1px solid var(--ln);background:var(--card);color:var(--fg);border-radius:20px;padding:6px 14px;cursor:pointer;font:inherit}
.seg button.on{background:#e50914;border-color:#e50914;color:#fff}
.p{display:none}.p.on{display:block}
.g{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.c{background:var(--card);border-radius:10px;padding:12px}.c h3{margin:0 0 8px;font-size:14px}
.cv{position:relative;height:270px}
.t{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--ln);vertical-align:top}th{color:var(--mut);font-weight:600}
select{font:inherit;padding:6px;border-radius:8px;border:1px solid var(--ln);background:var(--card);color:var(--fg)}
.n{font-size:13px;color:var(--mut)}li{margin:3px 0}
</style></head><body><main>
<header><h1>Netflix Content Strategy</h1>
<p class="m">End-to-end analysis, predictive models and business recommendations · {{n}} titles · additions through 25 Sep 2021</p></header>
<div class="k">
<div><b>{{n}}</b><span>titles analysed</span></div>
<div><b>{{mp}}% / {{tp}}%</b><span>movies / TV shows</span></div>
<div><b>{{tv20}}%</b><span>TV share of 2020 releases (was {{tv08}}% in 2008)</span></div>
<div><b>{{acc}}%</b><span>maturity classifier accuracy (baseline {{base}}%)</span></div>
<div><b>{{hitmeta}}%</b><span>recommender franchise hit@10, metadata only (random {{rnd}}%)</span></div></div>
<nav class="seg" id="tabs"><button data-t="s" class="on">Report</button><button data-t="e">Explore</button><button data-t="f">Forecast</button><button data-t="m">Models</button><button data-t="r">Recommender</button></nav>

<section id="s" class="p on">
<h2>Executive summary</h2>
<p>The catalogue is {{mp}}% movies and {{tp}}% TV shows, and the mix is shifting toward series: TV's share of releases rose from {{tv08}}% (2008) to {{tv20}}% (2020), while movie releases peaked in {{mpk}} and have fallen {{mdrop}}% since. Three models were built on title metadata. They are useful for triage and discovery but not yet for investment decisions, because the data holds no viewing or engagement signal.</p>
<h3>Key findings</h3><ul>
<li><b>Growth then plateau.</b> Titles added per year peaked in {{apk}} at {{apkn}}; 2021 is partial (to 25 Sep).</li>
<li><b>International content is the largest category.</b> "International Movies" ({{gim}} titles) and "International TV Shows" ({{git}}) are the most common genre tags.</li>
<li><b>Few Kids titles.</b> Kids content is {{kidm}}% of rated movies and {{kidt}}% of rated TV shows; Adults and Teens dominate.</li>
<li><b>Forecast.</b> TV releases are projected near {{tv25}} by 2025 (range {{tvlo}}-{{tvhi}}); movies stay flat near {{mv25}}. The total is unreliable: {{tot25}} from its own model vs {{bu25}} when the parts are added.</li>
<li><b>Models.</b> A {{model}} predicts maturity group with {{acc}}% accuracy (macro-F1 {{f1}}) vs a {{base}}% baseline, weakest on Teens ({{teen}}% recall). The recommender finds same-franchise titles in its top 10 for {{hitmeta}}% of titles using metadata alone.</li></ul>
<h3>Recommendations</h3>
<div class="t"><table><tr><th>Recommendation</th><th>Evidence</th><th>Confidence / next step</th></tr>
<tr><td>Prioritise series development</td><td>TV share {{tv08}}% to {{tv20}}%; TV is the only trend that beat a naive baseline in backtesting ({{tvmape}}% error).</td><td>Medium. Check retention and cost per title before moving budget.</td></tr>
<tr><td>Review the movie slate</td><td>Movie releases down {{mdrop}}% from the {{mpk}} peak.</td><td>Low-Medium. Recent years are likely under-counted; compare with production records.</td></tr>
<tr><td>Treat international content as a core pillar</td><td>International tags lead both movies and TV. Use genre tags, not country counts.</td><td>Medium. Validate with regional viewing data.</td></tr>
<tr><td>Fix metadata before more modelling</td><td>{{dirm}}% of titles lack a director; some country labels are wrong (e.g. Squid Game tagged Pakistan).</td><td>High. Add plot descriptions, cast and verified country.</td></tr>
<tr><td>Ship the content recommender as a cold-start fallback</td><td>{{hitmeta}}% franchise hit@10 (metadata only) vs {{rnd}}% random; it never crosses type and a few hub titles dominate.</td><td>Medium. Blend with viewing data; add cross-type and diversity rules.</td></tr>
<tr><td>Use the maturity model for QA, not auto-rating</td><td>{{acc}}% accuracy; Teens recall only {{teen}}%.</td><td>Medium. Flag likely mislabelled titles for human review.</td></tr>
<tr><td>Test a family / kids push</td><td>Kids is a small share of rated titles ({{kidm}}% movies, {{kidt}}% TV).</td><td>Low. Hypothesis only; validate with household viewing.</td></tr></table></div>
<h3>Approach</h3>
<p>Raw data was cleaned (placeholders to nulls, near-duplicates removed, dates and durations parsed), explored, then modelled: a TF-IDF content recommender evaluated by franchise recovery, a rolling-origin backtested release-year forecast, and a maturity classifier with leakage controls and train-only feature selection. All numbers on this page are computed from those outputs.</p>
<h3>Limitations</h3><ul>
<li>This is a catalogue snapshot, not viewership: nothing here shows what people watch or like.</li>
<li>2021 is partial, and recent release years are probably under-counted because titles keep arriving after release.</li>
<li>Country labels are unreliable, so no country conclusions are drawn.</li>
<li>The forecast rests on 13 yearly points and a trend break; treat it as a scenario.</li>
<li>Movie vs TV Show classification was set aside: genre labels give the answer away and "no director" alone scores 95%.</li></ul>
</section>

<section id="e" class="p"><h2>Explore the catalogue</h2>
<div class="seg" id="tf"><button data-v="All" class="on">All</button><button data-v="Movie">Movies</button><button data-v="TV Show">TV shows</button></div>
<div class="g"><div class="c"><h3>Titles by release year (* partial)</h3><div class="cv"><canvas id="c1"></canvas></div></div>
<div class="c"><h3>Top genre tags</h3><div class="cv"><canvas id="c2"></canvas></div></div>
<div class="c"><h3>Maturity group (rated titles)</h3><div class="cv"><canvas id="c3"></canvas></div></div></div></section>

<section id="f" class="p"><h2>Forecast of titles by release year</h2>
<div class="seg" id="sf"><button data-v="Total">Total</button><button data-v="Movie">Movies</button><button data-v="TV Show" class="on">TV shows</button></div>
<div class="c"><div class="cv" style="height:340px"><canvas id="c4"></canvas></div></div>
<p class="n">Best model per series chosen by backtesting. The shaded range is approximate and based on only a few backtest errors. Hollow point = 2021 titles seen so far.</p></section>

<section id="m" class="p"><h2>Model results</h2><div class="g">
<div class="c"><h3>Maturity classifier (test set)</h3><div class="cv"><canvas id="c5"></canvas></div></div>
<div class="c"><h3>Recommender: franchise hit rate @10</h3><div class="cv"><canvas id="c6"></canvas></div></div></div>
<p class="n">The top three classifiers are statistically tied. The last recommender bar adds title words, which also define the franchises, so it is partly circular; the bar above it is the fair test.</p></section>

<section id="r" class="p"><h2>Because you watched...</h2>
<p><select id="rq" aria-label="Pick a title"></select></p>
<div class="t"><table id="rt"></table></div>
<p class="n">Top 10 content-based matches (genres, director, rating, era, title words).</p></section>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const D=/*DATA*/null,$=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const v=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim(),R='#e50914',ch={};
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function mk(id,cfg){if(!window.Chart)return;ch[id]&&ch[id].destroy();Chart.defaults.color=v('--mut');Chart.defaults.borderColor=v('--ln');
cfg.options=Object.assign({responsive:true,maintainAspectRatio:false},cfg.options);ch[id]=new Chart($('#'+id),cfg)}
function seg(sel,fn){$$(sel+' button').forEach(b=>b.onclick=()=>{$$(sel+' button').forEach(x=>x.classList.toggle('on',x===b));fn(b.dataset.v)})}
let T='All',S='TV Show';
function ex(){const y=D.years,ink=v('--ink');
const ds=[['Movie','M',R],['TV Show','V',ink]].filter(a=>T=='All'||a[0]==T).map(a=>({label:a[0],data:y.map(r=>r[a[1]]),backgroundColor:a[2]}));
mk('c1',{type:'bar',data:{labels:y.map(r=>r.y),datasets:ds},options:{scales:{x:{stacked:true},y:{stacked:true}}}});
const g=D.genres[T];
mk('c2',{type:'bar',data:{labels:g.map(a=>a[0]),datasets:[{data:g.map(a=>a[1]),backgroundColor:R}]},options:{indexAxis:'y',plugins:{legend:{display:false}}}});
const a=D.aud[T];
mk('c3',{type:'doughnut',data:{labels:Object.keys(a),datasets:[{data:Object.values(a),backgroundColor:['#46a3ff','#f5a623',R],borderColor:v('--card')}]}})}
function fcst(){const f=D.fc[S],L=D.fy,n=f.hist.length,k=Array(n-1).fill(null),last=f.hist[n-1];
const ds=[{label:'lower',data:k.concat([last],f.lo),borderWidth:0,pointRadius:0},
{label:'Approx. 95% range',data:k.concat([last],f.hi),borderWidth:0,pointRadius:0,fill:'-1',backgroundColor:'rgba(229,9,20,.15)'},
{label:'Actual',data:f.hist.concat(Array(L.length-n).fill(null)),borderColor:v('--fg'),backgroundColor:v('--fg'),tension:.2},
{label:f.model+' forecast',data:k.concat([last],f.pred),borderColor:R,backgroundColor:R,borderDash:[6,4]},
{label:'2021 so far (partial)',data:Array(n).fill(null).concat([f.part],Array(L.length-n-1).fill(null)),showLine:false,pointRadius:6,pointStyle:'circle',backgroundColor:v('--bg'),borderColor:v('--fg')}];
if(f.bu)ds.push({label:'Movie + TV models',data:k.concat([last],f.bu),borderColor:v('--mut'),backgroundColor:v('--mut'),borderDash:[2,3],pointRadius:2});
mk('c4',{type:'line',data:{labels:L,datasets:ds},options:{plugins:{legend:{labels:{filter:i=>i.text!='lower'}}},scales:{y:{beginAtZero:true}}}})}
function mod(){const m=D.clf;
mk('c5',{type:'bar',data:{labels:m.map(r=>r[0]),datasets:[{label:'Accuracy',data:m.map(r=>r[1]),backgroundColor:v('--ink')},{label:'Macro-F1',data:m.map(r=>r[2]),backgroundColor:R}]},options:{scales:{y:{min:0,max:1}}}});
const e=D.rec;
mk('c6',{type:'bar',data:{labels:e.map(r=>r[0]),datasets:[{data:e.map(r=>r[1]),backgroundColor:R}]},options:{indexAxis:'y',scales:{x:{min:0,max:1}},plugins:{legend:{display:false}}}})}
function rc(){const r=D.recs[$('#rq').value]||[];
$('#rt').innerHTML='<tr><th>#</th><th>Title</th><th>Type</th><th>Year</th><th>Similarity</th></tr>'+r.map((x,i)=>`<tr><td>${i+1}</td><td>${esc(x[0])}</td><td>${esc(x[1])}</td><td>${x[2]}</td><td>${x[3].toFixed(2)}</td></tr>`).join('')}
const draw={e:ex,f:fcst,m:mod,r:rc};
$('#rq').innerHTML=Object.keys(D.recs).map(k=>`<option>${esc(k)}</option>`).join('');
$$('#tabs button').forEach(b=>b.onclick=()=>{$$('#tabs button').forEach(x=>x.classList.toggle('on',x===b));$$('.p').forEach(p=>p.classList.toggle('on',p.id==b.dataset.t));draw[b.dataset.t]&&draw[b.dataset.t]()});
seg('#tf',x=>{T=x;ex()});seg('#sf',x=>{S=x;fcst()});$('#rq').onchange=rc;
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{const t=$('#tabs .on').dataset.t;draw[t]&&draw[t]()});
</script></body></html>