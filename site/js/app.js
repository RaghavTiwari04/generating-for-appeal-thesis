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

  /* Human designed or generated?
   *
   * A game, and framed as one. The thesis never tested whether people can
   * tell the difference; it tested whether automated judges scored the two
   * groups similarly. Those are not the same claim and the intro says so
   * before anyone plays.
   *
   * Rounds come from data/quiz.json, which is filled from the evaluated set
   * once the gallery has been exported. With no rounds the section says it is
   * waiting rather than inventing anything.
   */
  Alpine.data('quiz', () => ({
    rounds: [],
    loaded: false,
    started: false,
    index: 0,
    answer: null,
    score: 0,

    async init() {
      try {
        // Deliberately not force-cache. This file gains rounds when a new
        // gallery is curated, and a visitor holding the older copy would sit
        // on an empty quiz with no way to find out otherwise.
        const r = await fetch('data/quiz.json');
        if (r.ok) {
          const doc = await r.json();
          this.rounds = Array.isArray(doc.rounds) ? doc.rounds : [];
        }
      } catch (e) {
        this.rounds = [];
      }
      this.loaded = true;
    },

    get ready() { return this.loaded && this.rounds.length > 0; },
    get round() { return this.rounds[this.index] || null; },
    get finished() { return this.index >= this.rounds.length; },
    get isHuman() { return this.round && this.round.condition === 'human'; },

    guess(said) {
      if (this.answer) return;
      this.answer = said;
      if ((said === 'human') === this.isHuman) this.score += 1;
    },

    next() {
      this.answer = null;
      this.index += 1;
    },

    /* The curated file alternates generated and marketplace so it can be read
     * and checked by hand. Played in that order the pattern is obvious after
     * three cards, so the order is thrown here instead, once per attempt. */
    shuffle() {
      const r = this.rounds.slice();
      for (let i = r.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [r[i], r[j]] = [r[j], r[i]];
      }
      this.rounds = r;
    },

    restart() {
      this.shuffle();
      this.index = 0;
      this.answer = null;
      this.score = 0;
      this.started = true;
    },
  }));

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

    // Reveal state. Nothing here exists until a choice has been committed.
    ranking: null,
    modelTop: null,
    opened: false,
    editedMessage: '',
    briefOpen: false,

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
        // Same reason as the quiz file: this one is filled in from a real
        // run later, and a cached empty copy would never correct itself.
        const r = await fetch('data/sample_run.json');
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

    /* Where the visitor's pick came in the model's order. */
    get chosenRank() {
      if (!this.ranking) return null;
      const row = this.ranking.find((r) => r.display_id === this.selected);
      return row ? row.rank : null;
    },

    get agreed() { return this.chosenRank === 1; },

    /* The cards in the model's order, each carrying the position it was
     * actually shown in, so the reveal can point at the grid the visitor
     * saw rather than at an abstract ranking. */
    get rankedCards() {
      if (!this.ranking) return [];
      return this.ranking
        .map((r) => ({
          ...r,
          card: this.cards.find((c) => c.display_id === r.display_id),
          shownAt: this.cards.findIndex((c) => c.display_id === r.display_id) + 1,
          isChoice: r.display_id === this.selected,
        }))
        .sort((a, b) => a.rank - b.rank);
    },

    /* Purchase intent is what the whole thesis ranks on, so it is the number
     * the bars show. Calibrated where the predictor supplies it. */
    scoreOf(row) {
      const s = row.scores || {};
      const v = s.purchase_intent_calibrated ?? s.purchase_intent ?? 0;
      return Number(v);
    },

    /* Bars are scaled to the batch, not to the full 0 to 1 range. Across four
     * siblings from one brief the absolute numbers sit within a few
     * hundredths of each other, and a bar chart from zero would show four
     * identical bars, which hides the very thing worth seeing. The caption
     * next to it says so, so nobody reads the widths as large differences. */
    barWidth(row) {
      const vals = this.rankedCards.map((r) => this.scoreOf(r));
      const lo = Math.min(...vals);
      const hi = Math.max(...vals);
      if (hi - lo < 1e-9) return 100;
      return 18 + 82 * ((this.scoreOf(row) - lo) / (hi - lo));
    },

    /* Committing is what unlocks the ranking. Until this runs, the browser
     * has never been told which card the model preferred. */
    async commit() {
      if (!this.selected) return;
      const elapsed = Math.round(performance.now() - this.gridShownAt);

      if (this.jobId) {
        // Fire and forget. A visitor should never wait on the study, and the
        // endpoint answers 204 whether or not it recorded anything.
        this.report('choice', elapsed);

        try {
          const r = await fetch(gcUrl(`/api/generate/${this.jobId}/reveal`), {
            method: 'POST',
            headers: gcHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ chosen_display_id: this.selected }),
          });
          if (!r.ok) throw new Error(`reveal failed (${r.status})`);
          const body = await r.json();
          this.ranking = body.candidates;
          this.modelTop = body.model_top_display_id;
        } catch (e) {
          // The choice still stands even if the comparison cannot be fetched.
          this.ranking = null;
          this.modelTop = null;
        }
      } else if (this._sample && this._sample.reveal) {
        this.ranking = this._sample.reveal.candidates;
        this.modelTop = this._sample.reveal.model_top_display_id;
      }

      this.editedMessage = this.chosenCard ? this.chosenCard.inside_message : '';
      this.opened = false;
      this.phase = 'reveal';
    },

    /* One place for every interaction the study records. Silent, optional,
     * and never blocking: if the backend is not logging, the endpoint answers
     * 204 and nothing here can tell the difference. */
    report(eventType, ms) {
      if (!this.jobId) return;
      fetch(gcUrl('/api/choice'), {
        method: 'POST',
        headers: gcHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({
          session_id: gcSession(),
          job_id: this.jobId,
          event_type: eventType,
          time_to_choice_ms: ms ?? null,
        }),
      }).catch(() => {});
    },

    open() {
      this.opened = true;
      this.phase = 'open';
    },

    /* The image is already a data URL at print resolution, 1240 by 1748,
     * which is A6 at 300 dpi. So the download is an anchor and nothing has to
     * be fetched: it keeps working after the job has aged out of the server's
     * memory. */
    downloadFront() {
      const card = this.chosenCard;
      if (!card) return;
      const a = document.createElement('a');
      a.href = card.image_data_url;
      a.download = `${(card.headline || 'card').toLowerCase().replace(/[^a-z0-9]+/g, '-')}.jpg`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      this.report('download_front');
    },

    messageEdited() { this.report('message_edited'); },

    startOver() {
      if (this.jobId) this.report('regenerate');
      this.phase = 'intro';
      this.cards = [];
      this.selected = null;
      this.jobId = null;
      this.ranking = null;
      this.modelTop = null;
      this.opened = false;
      this.briefOpen = false;
      this.replaying = false;
      this.error = '';
      this.resetStages();
    },
  }));
});
