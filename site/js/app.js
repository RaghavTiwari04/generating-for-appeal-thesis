/* Generating for Appeal: the card generator on the front of the site.
 *
 * One component drives the whole first screen. It moves through phases in
 * place rather than navigating, so the panel keeps its footprint and the
 * page never jumps under the reader:
 *
 *   intro -> generating -> choose -> reveal -> open
 *
 * The generator runs on a rented GPU that is usually off. Rather than showing
 * a dead form, the offline path replays a real captured run through the same
 * phases, so a visitor who arrives on a quiet day still sees what the system
 * does. Same component, same markup, different source of cards.
 */

document.addEventListener('alpine:init', () => {

  /* Is the backend up, and is it recording choices?
   *
   * Both answers come from /api/health, which is the one unauthenticated
   * route. Sections read this rather than each testing the network, and the
   * data notice is shown only when the backend says it is actually logging,
   * so the notice and the recording can never disagree.
   */
  Alpine.store('backend', {
    state: 'unknown',   // unknown | up | down
    logging: false,

    async check() {
      if (!window.GC_API && location.protocol !== 'file:') {
        // No backend address given. That is the normal state of the public
        // site: everything works except live generation.
        this.state = 'down';
        return false;
      }
      try {
        const r = await fetch(gcUrl('/api/health'), { cache: 'no-store' });
        if (!r.ok) throw new Error(r.status);
        const body = await r.json();
        this.state = 'up';
        this.logging = Boolean(body.logging);
        return true;
      } catch (e) {
        this.state = 'down';
        this.logging = false;
        return false;
      }
    },
  });

  Alpine.data('heroLoop', () => ({

    phase: 'intro',
    error: '',

    /* What the visitor tells us. Occasion is derived from this rather than
     * asked for: "who is it for" is a question a person can answer, and
     * "birthday/relationship" is not. */
    form: {
      relationship: '',
      tone: '',
      age: null,
    },

    /* Every value here exists in common/occasions.py. The tone list is the
     * five the brief prompt actually offers; formal-sincere and religious are
     * in the taxonomy but the prompt never presents them, so pinning one
     * would ask for a register the generator was not shown. */
    recipients: [
      { value: '', label: 'Someone, anyone' },
      { value: 'friend', label: 'A friend' },
      { value: 'partner', label: 'My partner' },
      { value: 'mum', label: 'My mum' },
      { value: 'dad', label: 'My dad' },
      { value: 'sister', label: 'My sister' },
      { value: 'brother', label: 'My brother' },
      { value: 'grandparent', label: 'A grandparent' },
      { value: 'child', label: 'A child' },
      { value: 'colleague', label: 'A colleague' },
    ],
    tones: [
      { value: '', label: 'Choose for me' },
      { value: 'warm-humorous', label: 'Funny' },
      { value: 'warm-sincere', label: 'Warm' },
      { value: 'funny-irreverent', label: 'Irreverent' },
      { value: 'sentimental', label: 'Sentimental' },
      { value: 'minimalist', label: 'Understated' },
    ],
    // MILESTONE_AGES, verbatim.
    milestones: [18, 21, 30, 40, 50, 60, 70, 80, 90, 100],

    // Generation state.
    jobId: null,
    cards: [],
    stages: [],
    elapsed: 0,
    _timer: null,
    _stream: null,
    replaying: false,

    // Choice state.
    selected: null,
    gridShownAt: 0,

    init() {
      this.resetStages();
      Alpine.store('backend').check();
    },

    /* birthday/kids, birthday/milestone, birthday/relationship or
     * birthday/general. The classifier reads listing titles to assign these,
     * and birthday/relationship means a romantic partner specifically, which
     * is why family recipients fall through to general. */
    get occasion() {
      if (this.form.age) return 'birthday/milestone';
      if (this.form.relationship === 'child') return 'birthday/kids';
      if (this.form.relationship === 'partner') return 'birthday/relationship';
      return 'birthday/general';
    },

    get liveBackend() { return Alpine.store('backend').state === 'up'; },

    get ctaLabel() {
      if (this.liveBackend) return 'Make four cards';
      return 'Watch a run that already happened';
    },

    pickAge(age) {
      this.form.age = this.form.age === age ? null : age;
    },

    resetStages() {
      this.stages = [
        {
          key: 'brief',
          name: 'Reading the market',
          detail: 'Writing four different briefs, each anchored on a card that already sells.',
          state: 'waiting',
        },
        {
          key: 'images',
          name: 'Drawing and lettering',
          detail: 'The greeting is painted into the artwork, not laid over it.',
          state: 'waiting',
        },
        {
          key: 'score',
          name: 'Scoring the batch',
          detail: 'The appeal model rates all four. You will not see how until you have chosen.',
          state: 'waiting',
        },
      ];
    },

    markStage(key, state) {
      const i = this.stages.findIndex((s) => s.key === key);
      if (i < 0) return;
      for (let j = 0; j < i; j++) this.stages[j].state = 'done';
      this.stages[i].state = state;
    },

    /* The stream sends prose written for a developer log, so the stage strip
     * matches on prefixes rather than expecting structured events. Anything
     * unrecognised just leaves the strip where it was. */
    readProgress(message) {
      const m = message.toLowerCase();
      if (m.includes('brief ready') || m.includes('cover images')) this.markStage('images', 'active');
      else if (m.includes('creative brief')) this.markStage('brief', 'active');
      else if (m.includes('scoring')) this.markStage('score', 'active');
      else if (m.startsWith('done')) this.stages.forEach((s) => { s.state = 'done'; });
    },

    startClock() {
      this.elapsed = 0;
      clearInterval(this._timer);
      this._timer = setInterval(() => { this.elapsed += 1; }, 1000);
    },

    stopClock() { clearInterval(this._timer); this._timer = null; },

    async begin() {
      this.error = '';
      this.resetStages();
      this.selected = null;

      const live = await Alpine.store('backend').check();
      if (!live) return this.replaySample();

      this.phase = 'generating';
      this.startClock();
      this.markStage('brief', 'active');

      try {
        const r = await fetch(gcUrl('/api/generate'), {
          method: 'POST',
          headers: gcHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({
            occasion: this.occasion,
            relationship: this.form.relationship || null,
            tone: this.form.tone || null,
            // Four, not the eight the experiments used. Four is the grid, and
            // the thesis found the extra four were not buying much appeal.
            n_candidates: 4,
            top_k: 4,
            constraints: this.form.age ? { age: this.form.age } : {},
          }),
        });
        if (r.status === 429) {
          const body = await r.json().catch(() => ({}));
          throw new Error(body.detail || 'The demo has reached its hourly limit.');
        }
        if (!r.ok) throw new Error(`The generator refused that request (${r.status}).`);
        const { job_id } = await r.json();
        this.jobId = job_id;
        this.listen(job_id);
      } catch (e) {
        this.stopClock();
        this.fail(e);
      }
    },

    listen(jobId) {
      this._stream = new EventSource(gcStreamUrl(`/api/generate/${jobId}`));

      this._stream.onmessage = (ev) => {
        const data = JSON.parse(ev.data);

        if (data.type === 'progress') {
          this.readProgress(data.message);
        } else if (data.type === 'done') {
          this._stream.close();
          this.stopClock();
          // Already shuffled server side, with rank and scores removed. The
          // order here carries no information about what the model preferred.
          this.cards = data.results;
          this.toChoose();
        } else if (data.type === 'error') {
          this._stream.close();
          this.stopClock();
          this.fail(new Error(data.message));
        }
      };

      this._stream.onerror = () => {
        this._stream.close();
        this.stopClock();
        this.fail(new Error('Lost the connection to the generator.'));
      };
    },

    toChoose() {
      this.phase = 'choose';
      // Started when the grid appears, so the measure is deliberation and not
      // however long the GPU took.
      this.gridShownAt = performance.now();
    },

    fail(e) {
      this.error = String(e.message || e);
      this.phase = 'intro';
    },

    /* The offline path. A real run, captured once and replayed through the
     * same phases at roughly the pace the real thing takes, so the page is
     * honest about what it is showing without being dead. */
    async replaySample() {
      this.replaying = true;
      this.phase = 'generating';
      this.startClock();

      try {
        const r = await fetch('data/sample_run.json', { cache: 'force-cache' });
        if (!r.ok) throw new Error('missing sample');
        const sample = await r.json();

        const beat = (ms) => new Promise((res) => setTimeout(res, ms));
        this.markStage('brief', 'active');
        await beat(1400);
        this.markStage('images', 'active');
        await beat(2600);
        this.markStage('score', 'active');
        await beat(1100);
        this.stages.forEach((s) => { s.state = 'done'; });

        this.jobId = null;
        this.cards = sample.cards;
        this._sample = sample;
        this.stopClock();
        this.toChoose();
      } catch (e) {
        this.stopClock();
        this.replaying = false;
        this.fail(new Error('The generator is offline and the sample run could not be loaded.'));
      }
    },

    select(card) {
      this.selected = this.selected === card.display_id ? null : card.display_id;
    },

    get chosenCard() {
      return this.cards.find((c) => c.display_id === this.selected) || null;
    },

    /* Committing is what unlocks the ranking. Until this runs, the browser
     * has never been told which card the model preferred. */
    async commit() {
      if (!this.selected) return;
      const elapsed = Math.round(performance.now() - this.gridShownAt);

      if (this.jobId) {
        // Fire and forget. A visitor should never wait on the study, and the
        // endpoint answers 204 whether or not it recorded anything.
        fetch(gcUrl('/api/choice'), {
          method: 'POST',
          headers: gcHeaders({ 'Content-Type': 'application/json' }),
          body: JSON.stringify({
            session_id: gcSession(),
            job_id: this.jobId,
            event_type: 'choice',
            time_to_choice_ms: elapsed,
          }),
        }).catch(() => {});
      }

      this.phase = 'reveal';
    },

    startOver() {
      this.phase = 'intro';
      this.cards = [];
      this.selected = null;
      this.jobId = null;
      this.replaying = false;
      this.error = '';
      this.resetStages();
    },
  }));
});
