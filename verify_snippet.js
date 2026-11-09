// Test harness, not part of the page. Paste into the DevTools console on index.html (served from the
// folder that also holds verify_expected.json). It pins --seed, lets the stylesheet compute the
// sprite, and compares the 768 final registers with the Python integer sampler.
(async () => {
  const { last, rows } = await (await fetch('verify_expected.json')).json();
  const app = document.querySelector('.app');
  const net = document.querySelector('.net');
  app.style.animation = 'none';
  document.getElementById('go').checked = true;
  let ok = 0;
  const bad = [];
  const t0 = performance.now();
  for (const [seed, want] of rows) {
    app.style.setProperty('--seed', seed);
    const cs = getComputedStyle(net);
    let acc = 0;
    for (let i = 0; i < 768; i++) {
      acc = (acc + (i + 1) * (parseInt(cs.getPropertyValue('--' + last + i)) + 100000)) % 1000000007;
    }
    if (acc === want) ok++; else bad.push(seed);
  }
  const ms = Math.round((performance.now() - t0) / rows.length);
  console.log(`${ok}/${rows.length} seeds match the reference, ${ms} ms per sprite`, bad);
  return { ok, total: rows.length, bad, msPerSprite: ms };
})();
