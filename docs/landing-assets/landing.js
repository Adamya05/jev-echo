(()=>{'use strict';
const $=id=>document.getElementById(id),poster=$('game').getContext('2d'),board=$('arena-board').getContext('2d'),arena=$('arena');
const RECORDED=[['Jev',5],['Echo',5],['Echo + search',41]],LIMIT=2000,TOUCH=matchMedia('(hover:none)').matches;
let game=SnakeGame(0),open=false,running=false,started=false,timer=null,nextMove=game.direction,touch=null,best=null,pushed=false;

function draw(ctx){const size=500,cell=size/10;ctx.clearRect(0,0,size,size);ctx.fillStyle='#eaede5';ctx.fillRect(0,0,size,size);ctx.strokeStyle='#dce2d6';ctx.lineWidth=1;for(let i=1;i<10;i++){ctx.beginPath();ctx.moveTo(i*cell,0);ctx.lineTo(i*cell,size);ctx.moveTo(0,i*cell);ctx.lineTo(size,i*cell);ctx.stroke()}game.body.forEach((p,i)=>{ctx.fillStyle=i?'#98a78d':'#416650';ctx.beginPath();ctx.roundRect(p[0]*cell+4,p[1]*cell+4,cell-8,cell-8,5);ctx.fill()});ctx.fillStyle='#ad8544';ctx.beginPath();ctx.arc((game.food[0]+.5)*cell,(game.food[1]+.5)*cell,9,0,Math.PI*2);ctx.fill();if(!game.alive){let h=game.body[0];ctx.strokeStyle='#ae5848';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(h[0]*cell+13,h[1]*cell+13);ctx.lineTo(h[0]*cell+37,h[1]*cell+37);ctx.moveTo(h[0]*cell+37,h[1]*cell+13);ctx.lineTo(h[0]*cell+13,h[1]*cell+37);ctx.stroke()}}
function rows(el,you){const list=RECORDED.map(([n,s])=>[n,s,false]);if(you!==null)list.push(['You',you,true]);list.sort((a,b)=>b[1]-a[1]);el.innerHTML=list.map(([n,s,y])=>`<div${y?' class="you"':''}><span>${n}</span><strong>${s}</strong></div>`).join('')}
function hint(text){$('arena-hint').textContent=text;$('arena-hint').style.opacity=text?1:0}
function score(){$('arena-score').textContent=game.score;rows($('arena-rows'),game.score)}
const over=()=>arena.classList.contains('over');

function fresh(){clearInterval(timer);timer=null;running=false;started=false;game=SnakeGame(0);nextMove=game.direction;arena.classList.remove('over');$('arena-end').hidden=true;draw(board);score();hint(TOUCH?'Swipe to start.':'Arrow keys or WASD to start.')}
function run(){started=true;hint('');if(!running){running=true;clearInterval(timer);timer=setInterval(tick,250)}}
function pause(why){if(!running)return;running=false;clearInterval(timer);timer=null;hint(why)}
function tick(){if(!running||!game.alive)return;game.step(nextMove);draw(board);score();if(!game.alive||game.steps>=LIMIT)finish()}
function finish(){running=false;clearInterval(timer);timer=null;best=best===null?game.score:Math.max(best,game.score);$('arena-end-title').textContent=game.alive?'Move limit reached.':'Game over.';$('arena-end-score').textContent=game.score;rows($('arena-end-rows'),game.score);arena.classList.add('over');$('arena-end').hidden=false;$('arena-again').focus({preventScroll:true})}
function move(dir){if(!open||over())return;if(!running)run();if(dir===OPP[game.direction])return;nextMove=dir}

function show(){open=true;arena.hidden=false;document.body.classList.add('arena-open');if(!pushed){history.pushState({arena:1},'');pushed=true}fresh();$('arena-exit').focus({preventScroll:true})}
function hide(viaHistory){if(!open)return;open=false;clearInterval(timer);timer=null;running=false;arena.hidden=true;document.body.classList.remove('arena-open');if(best!==null){$('last-score').textContent=`You, best · ${best}`;$('your-best').textContent=best;$('your-best-row').hidden=false;$('play-button').innerHTML='Play again <span>→</span>'}$('play-button').focus({preventScroll:true});if(pushed&&!viaHistory){pushed=false;history.back()}else pushed=false}

$('play-button').addEventListener('click',show);$('arena-again').addEventListener('click',fresh);$('arena-exit').addEventListener('click',()=>hide(false));$('arena-exit-2').addEventListener('click',()=>hide(false));
window.addEventListener('popstate',()=>{if(open)hide(true)});
document.addEventListener('keydown',e=>{if(!open)return;if(e.key==='Escape'){e.preventDefault();hide(false);return}let m={ArrowUp:'UP',ArrowDown:'DOWN',ArrowLeft:'LEFT',ArrowRight:'RIGHT',w:'UP',a:'LEFT',s:'DOWN',d:'RIGHT',W:'UP',A:'LEFT',S:'DOWN',D:'RIGHT'}[e.key];if(m){e.preventDefault();move(m)}});
arena.addEventListener('touchstart',e=>{let t=e.touches[0];touch=[t.clientX,t.clientY]},{passive:true});
arena.addEventListener('touchmove',e=>{if(open)e.preventDefault()},{passive:false});
arena.addEventListener('touchend',e=>{if(!touch)return;let t=e.changedTouches[0],dx=t.clientX-touch[0],dy=t.clientY-touch[1];touch=null;if(e.target.closest('button'))return;if(Math.max(Math.abs(dx),Math.abs(dy))<12){if(open&&started&&!running&&!over())run();return}move(Math.abs(dx)>Math.abs(dy)?dx>0?'RIGHT':'LEFT':dy>0?'DOWN':'UP')},{passive:true});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&open)pause(TOUCH?'Paused. Tap to resume.':'Paused. Press a key to resume.')});

$('show-explainer').addEventListener('click',()=>{let o=$('explainer').hidden;$('explainer').hidden=!o;$('show-explainer').setAttribute('aria-expanded',o);$('show-explainer').innerHTML=o?'Close the explanation <span>↑</span>':'Explore one decision <span>↓</span>';if(o&&!$('explainer-frame').getAttribute('src'))$('explainer-frame').src='landing-assets/search-explainer.html'});
draw(poster);
})();
