/* Where the generator lives, and how to get in.
 *
 * The site is hosted permanently; the backend that generates cards runs on a
 * rented GPU and is usually switched off, because keeping FLUX resident costs
 * money by the hour. So the address and the token are handed over in the link
 * rather than baked in:
 *
 *   ?api=https://demo-api.example&token=...
 *
 * The token is a demo credential. It exists to cap spend and to keep the
 * generator off the open internet, not to identify anyone.
 *
 * Lifted from app/frontend/index.html, which has been using this since the
 * split-hosting work. Kept identical on purpose: two copies that drift would
 * mean the dev UI and the public site reach different backends.
 */
(function () {
  const q = new URLSearchParams(location.search);

  window.GC_API = (q.get('api') || window.GC_API_BASE || '').replace(/\/$/, '');
  window.GC_TOKEN = q.get('token') || localStorage.getItem('gc_token') || '';
  if (q.get('token')) localStorage.setItem('gc_token', q.get('token'));

  window.gcUrl = (path) => `${window.GC_API}${path}`;

  window.gcHeaders = (extra) => Object.assign(
    {}, extra || {}, window.GC_TOKEN ? { 'X-Access-Token': window.GC_TOKEN } : {});

  // EventSource cannot set headers, so the stream carries its token in the
  // query string. See app/auth.py for why that is scoped to this one route.
  window.gcStreamUrl = (path) => {
    const u = window.gcUrl(path);
    return window.GC_TOKEN
      ? `${u}${u.includes('?') ? '&' : '?'}token=${encodeURIComponent(window.GC_TOKEN)}`
      : u;
  };

  /* One id per browser tab, for the choice study.
   *
   * sessionStorage rather than localStorage: it dies when the tab closes, so
   * a person who comes back tomorrow is a new session and the two visits
   * cannot be joined. That is the intended limit, not an oversight. Nothing
   * is sent anywhere unless the backend has logging switched on.
   */
  window.gcSession = () => {
    let id = sessionStorage.getItem('gc_session');
    if (!id) {
      id = (crypto.randomUUID && crypto.randomUUID()) ||
           `${Date.now()}-${Math.random().toString(16).slice(2)}`;
      sessionStorage.setItem('gc_session', id);
    }
    return id;
  };
})();
