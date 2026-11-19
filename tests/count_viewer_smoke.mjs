// Verify viewer state and the Canvas call path without opening a browser.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const html = readFileSync(new URL('../src/vibrotrack/count_viewer.html', import.meta.url), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0];
const data = {
  image:'preview', full:'original', width:2401, height:2399,
  particles:[[100, 100, 31.7, 0], [180, 180, 36.9, 1]],
  meta:{image:'frame.tiff', mode:'bi12-14', mode_label:'12 + 14 mm 双分散', n:2, ms:201,
    neighbours:[], groups:[{id:0, radius:31.7, label:'12 mm'}, {id:1, radius:36.9, label:'14 mm'}]},
};
for (const inApp of [true, false]) {
  const edits = new Map(), drawCalls = [], labels = [];
  let exported, downloads = 0;
  class Element {
    constructor(tag='div') {
      this.tag = tag;
      this.style = {};
      this.value = '';
      this.children = [];
      this.clientWidth = 1000;
      this.clientHeight = 720;
      this.classList = {add(){}, remove(){}, toggle(){}};
      this.handlers = {};
    }
    addEventListener(type, fn) { this.handlers[type] = fn; }
    replaceChildren(...items) { this.children = items; }
    append(...items) { this.children.push(...items); }
    click() { if (this.tag === 'a') { downloads++; exported = this.download; } return this.onclick?.(); }
    toBlob(fn) { assert.equal(this.width, 2401); assert.equal(this.height, 2399); fn({}); }
    getBoundingClientRect() { return {left:0, top:0}; }
    getContext() { return new Proxy({}, {get(_, key) {
      if (key === 'drawImage') return (...args) => drawCalls.push(args);
      if (key === 'fillText') return s => labels.push(s);
      if (key === 'measureText') return () => ({width:200});
      return () => {};
    }}); }
  }
  let nextFrame = 0;
  const win = new Element();
  win.confirm = () => true;
  const c = vm.createContext({
    queueMicrotask, window:win, setTimeout:fn => fn(), URLSearchParams,
    URL:{createObjectURL:() => 'blob:test', revokeObjectURL(){}},
    document:{createElement:tag => new Element(tag), getElementById:id => edits.get(id)},
    requestAnimationFrame(fn) { queueMicrotask(fn); return ++nextFrame; },
    Option:class {constructor(label, value) {this.label=label; this.value=value;}},
    Image:class {set src(url) {
      Object.assign(this, {url, naturalWidth:url === 'preview' ? 1200 : 2401,
                            naturalHeight:url === 'preview' ? 1199 : 2399});
      queueMicrotask(() => this.onload());
    }},
    fetch:async (url, req) => {
      assert.equal(url, '/detect?mode=bi12-14&name=frame.tiff');
      assert.equal(req.method, 'POST');
      assert.equal(req.body.name, 'frame.tiff');
      return {ok:true, json:async () => structuredClone(data)};
    },
  });
  for (const m of html.matchAll(/id="([^"]+)"/g)) edits.set(m[1], new Element());
  vm.runInContext(script.replace('/*__DATA__*/', inApp ? 'null' : JSON.stringify(data)), c);
  if (inApp) {
    assert.equal(edits.get('workflow').disabled, true);
    edits.get('sizeMode').value = 'bi12-14';
    edits.get('sizeMode').onchange();
    assert.equal(edits.get('workflow').disabled, false);
    await vm.runInContext("detectFile({name:'frame.tiff', size:100})", c);
  }
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(edits.get('count').textContent, 2);
  assert.equal(edits.get('infoMode').textContent, data.meta.mode_label);
  const counts = () => edits.get('groups').children.map(el => el.children[1].textContent);
  assert.deepEqual(counts(), [1, 1]);
  assert.equal(drawCalls[0][0].url, 'preview');
  assert.ok(Math.abs(drawCalls[0][3] - 1200) < 1e-8);
  assert.ok(Math.abs(drawCalls[0][4] - 1199) < 1e-8);
  edits.get('modeAdd').click();
  edits.get('addGroup').value = '0';
  vm.runInContext('addAt(300, 300); removeAt(S.particles[1]);', c);
  assert.deepEqual(counts(), [2, 0]);
  vm.runInContext('undo()', c);
  assert.deepEqual(counts(), [2, 1]);
  await vm.runInContext('exportPng()', c);
  assert.equal(drawCalls.find(call => call[0].url === 'original')[3], 2401);
  assert.ok(labels.some(label => label.includes('12 mm 2') && label.includes('14 mm 1')));
  assert.equal(downloads, 1);
  assert.equal(exported, 'frame_count3.png');
  edits.get('addGroup').value = '';
  vm.runInContext('addAt(350, 350); addAt(-10, 350);', c);
  assert.deepEqual(counts(), [2, 1, 1]);
  assert.equal(edits.get('count').textContent, 4);
  vm.runInContext('removeAt(S.particles[0]); clearManual();', c);
  assert.deepEqual(counts(), [1, 1]);
}
console.log('PASS: local upload and offline bundle; class counts, undo/reset, original-size export');
