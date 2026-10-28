// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package swebench

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"
)

// Stat summarizes one kind of event.
type Stat struct {
	Event  string  `json:"event"`
	Name   string  `json:"name,omitempty"`
	Count  int     `json:"count"`
	Failed int     `json:"failed"`
	P50MS  float64 `json:"p50_ms"`
	P95MS  float64 `json:"p95_ms"`
	MaxMS  float64 `json:"max_ms"`
	// Retried counts ttfi and wake events that took more than one attempt:
	// the environment was not serving when first asked.
	Retried int `json:"retried,omitempty"`
}

// Unresolved is an instance whose reward step scored 0, with the grader's
// verdict when the step's output ended in one.
type Unresolved struct {
	Instance string `json:"instance"`
	Reason   string `json:"reason"`
}

// StepStat is one training step in steps mode.
type StepStat struct {
	Step     int `json:"step"`
	Tasks    int `json:"tasks"`
	Failed   int `json:"failed"`
	Rewarded int `json:"rewarded"`
	// Live is how many environments were still in the fleet at this step;
	// steps mode drops an environment after its first failure, so a
	// shrinking Live makes later steps' counts incomparable.
	Live      int     `json:"live"`
	BarrierMS float64 `json:"barrier_ms"`
	ParkMS    float64 `json:"park_ms"`
	WakeP50MS float64 `json:"wake_p50_ms"`
	WakeP95MS float64 `json:"wake_p95_ms"`

	WorkingSetHits       int     `json:"working_set_hits,omitempty"`
	WorkingSetMisses     int     `json:"working_set_misses,omitempty"`
	WorkingSetHitRate    float64 `json:"working_set_hit_rate,omitempty"`
	StandingPoolSize     int     `json:"standing_pool_size,omitempty"`
	Created              int     `json:"created,omitempty"`
	Reused               int     `json:"reused,omitempty"`
	Retired              int     `json:"retired,omitempty"`
	PrefetchLeadMarginMS float64 `json:"prefetch_lead_margin_ms,omitempty"`
	ReleaseOverlapMS     float64 `json:"release_overlap_ms,omitempty"`
}

// Report is the summary of one run's JSONL, stable enough to diff between
// runs.
type Report struct {
	RunID string `json:"run"`
	Mode  string `json:"mode,omitempty"`
	// Reps is how many runs the report pools, the run events it saw: one for
	// a file, the reps of an arm under report -arm (see Pool). Runs lists
	// their run ids in order.
	Reps       int        `json:"reps,omitempty"`
	Runs       []string   `json:"runs,omitempty"`
	RunMS      float64    `json:"run_ms,omitempty"`
	Tasks      int        `json:"tasks"`
	Failed     int        `json:"failed"`
	Rewarded   int        `json:"rewarded"`
	RewardRate float64    `json:"reward_rate"`
	Stats      []Stat     `json:"stats"`
	Steps      []StepStat `json:"steps,omitempty"`

	WorkingSetHits       int     `json:"working_set_hits,omitempty"`
	WorkingSetMisses     int     `json:"working_set_misses,omitempty"`
	WorkingSetHitRate    float64 `json:"working_set_hit_rate,omitempty"`
	StandingPoolSize     int     `json:"standing_pool_size,omitempty"`
	Created              int     `json:"created,omitempty"`
	Reused               int     `json:"reused,omitempty"`
	Retired              int     `json:"retired,omitempty"`
	PrefetchLeadMarginMS float64 `json:"prefetch_lead_margin_ms,omitempty"`
	ReleaseOverlapMS     float64 `json:"release_overlap_ms,omitempty"`
	// Errors counts distinct error strings, most frequent first.
	Errors []ErrorCount `json:"errors,omitempty"`
	// Unresolved lists the instances whose reward was 0, once each.
	Unresolved []Unresolved `json:"unresolved,omitempty"`
	// Cache is what the node image caches did between the samples given to
	// report -cache-before and -cache-after; nil without them.
	Cache *CacheDelta `json:"cache,omitempty"`
	// Restores is atelet's restore phase breakdown joined to the run's
	// environments (report -restore-log); nil without it.
	Restores *RestoreBreakdown `json:"restores,omitempty"`
	// Labels are the run event's labels: the loader's own and the operator's.
	Labels map[string]string `json:"labels,omitempty"`
	// Slowest lists the slowest first commands and wakes (report -top N).
	Slowest []SlowEvent `json:"slowest,omitempty"`
}

// SlowEvent is one of the slowest first commands or wakes of a run.
type SlowEvent struct {
	Event    string  `json:"event"`
	Env      string  `json:"env"`
	Instance string  `json:"instance,omitempty"`
	Step     int     `json:"step,omitempty"`
	DurMS    float64 `json:"dur_ms"`
	Attempts int     `json:"attempts,omitempty"`
}

// Slowest returns the n slowest successful ttfi events, then the n slowest
// successful wake events, longest first.
func Slowest(events []Event, n int) []SlowEvent {
	var out []SlowEvent
	for _, kind := range []string{EventTTFI, EventWake} {
		var picked []SlowEvent
		for _, e := range events {
			if e.Event == kind && e.OK {
				picked = append(picked, SlowEvent{Event: e.Event, Env: e.Env, Instance: e.Instance, Step: e.Step, DurMS: e.DurMS, Attempts: e.Attempts})
			}
		}
		sort.SliceStable(picked, func(i, j int) bool { return picked[i].DurMS > picked[j].DurMS })
		if len(picked) > n {
			picked = picked[:n]
		}
		out = append(out, picked...)
	}
	return out
}

// ErrorCount is one error string and how often it occurred.
type ErrorCount struct {
	Err   string `json:"err"`
	Count int    `json:"count"`
}

type statKey struct{ event, name string }

// Builder folds a stream of events into a Report without materializing the full
// event slice in memory.
type Builder struct {
	rep        Report
	durs       map[statKey][]float64
	failed     map[statKey]int
	retried    map[statKey]int
	unresolved map[string]string
	order      []statKey
	steps      map[int]*StepStat
	wakes      map[int][]float64
	barriers   map[int][]float64
	parks      map[int][]float64
	lives      map[int][]float64
	runs       []float64
	labels     labelUnion
	errs       map[string]int
}

// NewBuilder returns a streaming report Builder.
func NewBuilder() *Builder {
	return &Builder{
		durs:       map[statKey][]float64{},
		failed:     map[statKey]int{},
		retried:    map[statKey]int{},
		unresolved: map[string]string{},
		steps:      map[int]*StepStat{},
		wakes:      map[int][]float64{},
		barriers:   map[int][]float64{},
		parks:      map[int][]float64{},
		lives:      map[int][]float64{},
		labels:     labelUnion{},
		errs:       map[string]int{},
	}
}

// Add folds one event into the Builder.
func (b *Builder) Add(e Event) {
	if b.rep.RunID == "" {
		b.rep.RunID = e.RunID
	}
	k := statKey{e.Event, ""}
	if e.Event == EventStep || e.Event == EventPark || e.Event == EventTask {
		k.name = e.Name // a task's name is its trajectory, so a mix gets a row each
	}
	if _, ok := b.durs[k]; !ok {
		b.order = append(b.order, k)
		b.durs[k] = nil
	}
	if e.Attempts > 1 {
		b.retried[k]++
	}
	if e.Event == EventStep && e.Reward != nil && *e.Reward == 0 {
		if _, seen := b.unresolved[e.Instance]; !seen {
			if e.Detail != "" {
				b.unresolved[e.Instance] = e.Detail
			} else {
				b.unresolved[e.Instance] = verdict(e.Tail)
			}
		}
	}
	if e.OK {
		b.durs[k] = append(b.durs[k], e.DurMS)
	} else {
		b.failed[k]++
		if e.Err != "" {
			b.errs[e.Err]++
		}
	}
	switch e.Event {
	case EventRun:
		b.rep.Reps++
		b.rep.Runs = append(b.rep.Runs, e.RunID)
		if b.rep.Mode == "" {
			b.rep.Mode = e.Name
		}
		b.runs = append(b.runs, e.DurMS)
		b.labels.add(e.Labels)
		b.rep.WorkingSetHits = e.WorkingSetHits
		b.rep.WorkingSetMisses = e.WorkingSetMisses
		b.rep.WorkingSetHitRate = e.WorkingSetHitRate
		b.rep.StandingPoolSize = e.StandingPoolSize
		b.rep.Created = e.Created
		b.rep.Reused = e.Reused
		b.rep.Retired = e.Retired
		b.rep.PrefetchLeadMarginMS = e.PrefetchLeadMarginMS
		b.rep.ReleaseOverlapMS = e.ReleaseOverlapMS
	case EventTask:
		b.rep.Tasks++
		if !e.OK {
			b.rep.Failed++
		}
		if e.Reward != nil && *e.Reward == 1 {
			b.rep.Rewarded++
		}
		if e.Step > 0 {
			s := stepStat(b.steps, e.Step)
			s.Tasks++
			if !e.OK {
				s.Failed++
			}
			if e.Reward != nil && *e.Reward == 1 {
				s.Rewarded++
			}
		}
	case EventBarrier:
		st := stepStat(b.steps, e.Step)
		b.barriers[e.Step] = append(b.barriers[e.Step], e.DurMS)
		b.lives[e.Step] = append(b.lives[e.Step], float64(e.Count))
		st.WorkingSetHits = e.WorkingSetHits
		st.WorkingSetMisses = e.WorkingSetMisses
		st.WorkingSetHitRate = e.WorkingSetHitRate
		st.StandingPoolSize = e.StandingPoolSize
		st.Created = e.Created
		st.Reused = e.Reused
		st.Retired = e.Retired
		st.PrefetchLeadMarginMS = e.PrefetchLeadMarginMS
		st.ReleaseOverlapMS = e.ReleaseOverlapMS
	case EventPark:
		st := stepStat(b.steps, e.Step)
		b.parks[e.Step] = append(b.parks[e.Step], e.DurMS)
		if e.StandingPoolSize > 0 {
			st.StandingPoolSize = e.StandingPoolSize
		}
		if e.ReleaseOverlapMS > 0 {
			st.ReleaseOverlapMS = e.ReleaseOverlapMS
		}
	case EventWake:
		if e.OK {
			b.wakes[e.Step] = append(b.wakes[e.Step], e.DurMS)
		}
	}
}

// Finish computes exact percentiles over the accumulated durations and returns
// the finalized Report.
func (b *Builder) Finish() *Report {
	rep := b.rep
	if rep.Tasks > 0 {
		rep.RewardRate = float64(rep.Rewarded) / float64(rep.Tasks)
	}
	rep.RunMS = median(b.runs)
	rep.Labels = b.labels.merged()
	for _, k := range b.order {
		d := b.durs[k]
		sort.Float64s(d)
		rep.Stats = append(rep.Stats, Stat{
			Event:   k.event,
			Name:    k.name,
			Count:   len(d) + b.failed[k],
			Failed:  b.failed[k],
			P50MS:   percentile(d, 0.50),
			P95MS:   percentile(d, 0.95),
			MaxMS:   percentile(d, 1),
			Retried: b.retried[k],
		})
	}
	for step, s := range b.steps {
		w := b.wakes[step]
		sort.Float64s(w)
		cp := *s
		cp.WakeP50MS = percentile(w, 0.5)
		cp.WakeP95MS = percentile(w, 0.95)
		cp.BarrierMS = median(b.barriers[step])
		cp.ParkMS = median(b.parks[step])
		cp.Live = 0
		for _, n := range b.lives[step] {
			cp.Live += int(n)
		}
		rep.Steps = append(rep.Steps, cp)
	}
	sort.Slice(rep.Steps, func(i, j int) bool { return rep.Steps[i].Step < rep.Steps[j].Step })
	for msg, n := range b.errs {
		rep.Errors = append(rep.Errors, ErrorCount{Err: msg, Count: n})
	}
	for inst, why := range b.unresolved {
		rep.Unresolved = append(rep.Unresolved, Unresolved{Instance: inst, Reason: why})
	}
	sort.Slice(rep.Unresolved, func(i, j int) bool { return rep.Unresolved[i].Instance < rep.Unresolved[j].Instance })
	sort.Slice(rep.Errors, func(i, j int) bool {
		if rep.Errors[i].Count != rep.Errors[j].Count {
			return rep.Errors[i].Count > rep.Errors[j].Count
		}
		return rep.Errors[i].Err < rep.Errors[j].Err
	})
	return &rep
}

// Summarize streams JSONL events from r into a Builder and returns the report.
func Summarize(r io.Reader) (*Report, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<20), 16<<20)
	b := NewBuilder()
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var e Event
		if err := json.Unmarshal(line, &e); err != nil {
			return nil, fmt.Errorf("parse event: %w", err)
		}
		b.Add(e)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return b.Finish(), nil
}

// ReadEvents reads a JSONL events file.
func ReadEvents(r io.Reader) ([]Event, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<20), 16<<20)
	var events []Event
	for sc.Scan() {
		b := bytes.TrimSpace(sc.Bytes())
		if len(b) == 0 {
			continue
		}
		var e Event
		if err := json.Unmarshal(b, &e); err != nil {
			return nil, fmt.Errorf("parse event: %w", err)
		}
		events = append(events, e)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return events, nil
}

// Build computes the report from events in memory.
func Build(events []Event) *Report {
	b := NewBuilder()
	for _, e := range events {
		b.Add(e)
	}
	return b.Finish()
}

// verdict extracts the grader's verdict from the end of a reward step's
// output: "invalid run", "f2p A/B p2p C/D", or "reward 0" when the output
// does not end in the grader's JSON line.
func verdict(tail string) string {
	if a := AdapterFor(DefaultKind); a != nil && a.Grader() != nil {
		if _, detail, err := a.Grader().Parse([]byte(tail)); err == nil && detail != "" {
			return detail
		}
	}
	return "reward 0"
}

// WriteExclusions writes the unresolved instances in the exclude-file format:
// one id per line followed by the reason as a comment.
func (rep *Report) WriteExclusions(w io.Writer, header string) {
	fmt.Fprintf(w, "# %s\n# Generated from run %s: %d of %d tasks unresolved.\n", header, rep.RunID, len(rep.Unresolved), rep.Tasks)
	for _, u := range rep.Unresolved {
		fmt.Fprintf(w, "%s  %s\n", u.Instance, u.Reason)
	}
}

func stepStat(m map[int]*StepStat, step int) *StepStat {
	s, ok := m[step]
	if !ok {
		s = &StepStat{Step: step}
		m[step] = s
	}
	return s
}

// median sorts d and returns its middle value, or the mean of the two
// middle values for an even count, so two reps of an arm pool to the value
// between them rather than the lower one. 0 for no values.
func median(d []float64) float64 {
	n := len(d)
	if n == 0 {
		return 0
	}
	sort.Float64s(d)
	if n%2 == 1 {
		return d[n/2]
	}
	return (d[n/2-1] + d[n/2]) / 2
}

// labelUnion merges the labels of pooled runs' run events: a key keeps its
// value when every run agrees, else every value in the order first seen,
// "|"-joined, so a column that mixes commits says so.
type labelUnion map[string][]string

func (u labelUnion) add(labels map[string]string) {
	for k, v := range labels {
		if !slices.Contains(u[k], v) {
			u[k] = append(u[k], v)
		}
	}
}

func (u labelUnion) merged() map[string]string {
	if len(u) == 0 {
		return nil
	}
	out := make(map[string]string, len(u))
	for k, vs := range u {
		out[k] = strings.Join(vs, "|")
	}
	return out
}

// Pool builds one report over the reps of an arm, as Build over their merged
// events (counts add up, durations take the median, Runs lists the reps).
// The reps must be complete, distinct runs of one workload: each with a run
// event and the loader's labels, no run id twice, and poolKeys agreeing.
func Pool(runs [][]Event) (*Report, error) {
	var all []Event
	seen := map[string]int{}
	var shape map[string]string
	for i, events := range runs {
		first := i == 0
		var run *Event
		n := 0
		for j := range events {
			if events[j].Event == EventRun {
				run = &events[j]
				n++
			}
		}
		if run == nil {
			return nil, fmt.Errorf("rep %d has no run event: an interrupted run cannot be pooled", i+1)
		}
		if n > 1 {
			return nil, fmt.Errorf("rep %d holds %d run events: one file is one run, give each run its own file", i+1, n)
		}
		if run.Labels["mode"] == "" {
			return nil, fmt.Errorf("rep %d (run %q) has no loader labels: pool only runs whose loader labeled the run event", i+1, run.RunID)
		}
		if prev, dup := seen[run.RunID]; dup {
			return nil, fmt.Errorf("rep %d repeats run id %q of rep %d", i+1, run.RunID, prev)
		}
		seen[run.RunID] = i + 1
		if first {
			shape = run.Labels
		} else {
			for _, k := range poolKeys(shape, run.Labels) {
				if shape[k] != run.Labels[k] {
					return nil, fmt.Errorf("rep %d differs from rep 1 in %s (%q, %q): pool only reps of one workload", i+1, k, run.Labels[k], shape[k])
				}
			}
		}
		all = append(all, events...)
	}
	return Build(all), nil
}

// poolKeys are the labels two reps must agree on: the loader's own and every
// -set value (set.KEY) either of them carries.
func poolKeys(a, b map[string]string) []string {
	var set []string
	for _, m := range []map[string]string{a, b} {
		for k := range m {
			if strings.HasPrefix(k, "set.") && !slices.Contains(set, k) {
				set = append(set, k)
			}
		}
	}
	sort.Strings(set)
	return append(slices.Clone(LoaderLabelKeys), set...)
}

// percentile of a sorted slice by the nearest-rank method, ceil(p*n)-1; 0
// for an empty slice.
func percentile(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if p >= 1 {
		return sorted[n-1]
	}
	i := int(math.Ceil(p*float64(n))) - 1
	if i < 0 {
		i = 0
	}
	if i >= n {
		i = n - 1
	}
	return sorted[i]
}

// WriteText prints the report as tables.
func (rep *Report) WriteText(w io.Writer) {
	fmt.Fprintf(w, "run %s mode=%s tasks=%d failed=%d rewarded=%d reward_rate=%.3f run=%.1fs\n",
		rep.RunID, rep.Mode, rep.Tasks, rep.Failed, rep.Rewarded, rep.RewardRate, rep.RunMS/1000)
	if len(rep.Labels) > 0 {
		fmt.Fprintf(w, "labels: %s\n", labelLine(rep.Labels))
	}
	fmt.Fprintln(w)
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "event\tname\tcount\tfailed\tretried\tp50_ms\tp95_ms\tmax_ms")
	for _, s := range rep.Stats {
		fmt.Fprintf(tw, "%s\t%s\t%d\t%d\t%d\t%.0f\t%.0f\t%.0f\n", s.Event, s.Name, s.Count, s.Failed, s.Retried, s.P50MS, s.P95MS, s.MaxMS)
	}
	tw.Flush()
	if len(rep.Steps) > 0 {
		fmt.Fprintln(w)
		tw = tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "step\tlive\ttasks\tfailed\trewarded\tbarrier_s\tpark_s\twake_p50_ms\twake_p95_ms")
		for _, s := range rep.Steps {
			fmt.Fprintf(tw, "%d\t%d\t%d\t%d\t%d\t%.1f\t%.1f\t%.0f\t%.0f\n", s.Step, s.Live, s.Tasks, s.Failed, s.Rewarded,
				s.BarrierMS/1000, s.ParkMS/1000, s.WakeP50MS, s.WakeP95MS)
		}
		tw.Flush()
	}
	if c := rep.Cache; c != nil {
		first := 0
		for _, s := range rep.Stats {
			if s.Event == EventTTFI {
				first = s.Count
			}
		}
		fmt.Fprintf(w, "\nimage cache during the run: %d nodes differenced", c.Nodes)
		for _, x := range []struct {
			n    int
			what string
		}{{c.Restarted, "restarted, counted from zero"}, {c.Added, "added, counted from zero"}, {c.Removed, "removed, left out"}, {c.Skipped, "with no before sample, left out"}} {
			if x.n > 0 {
				fmt.Fprintf(w, ", %d %s", x.n, x.what)
			}
		}
		fmt.Fprintf(w, "; misses (cold pulls) %.0f on %d nodes, hits %.0f, errors %.0f", c.Misses, c.NodesWithMisses, c.Hits, c.Errors)
		if first > 0 {
			fmt.Fprintf(w, "; first commands %d, misses per first command %.2f", first, c.Misses/float64(first))
		}
		fmt.Fprintln(w)
	}
	if b := rep.Restores; b != nil {
		fmt.Fprint(w, "\nrestore phases from atelet's log, paired with ttfi (first) and wake events by time; outside = the event's span minus atelet's total")
		for _, x := range []struct {
			n    int
			what string
		}{{b.OutsideRun, "records before or after this run"}, {b.Unmatched, "records for actors not in this run"}, {b.Unpaired, "records inside no ttfi or wake window"}} {
			if x.n > 0 {
				fmt.Fprintf(w, "; %d %s", x.n, x.what)
			}
		}
		fmt.Fprintln(w)
		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "restore\tphase\tcount\tp50_ms\tp95_ms\tmax_ms")
		group := func(label string, g RestoreGroup) {
			for _, p := range g.Phases {
				fmt.Fprintf(tw, "%s\t%s\t%d\t%.0f\t%.0f\t%.0f\n", label, p.Phase, p.Count, p.P50MS, p.P95MS, p.MaxMS)
			}
			if p := g.Outside; p != nil {
				fmt.Fprintf(tw, "%s\t%s\t%d\t%.0f\t%.0f\t%.0f\n", label, p.Phase, p.Count, p.P50MS, p.P95MS, p.MaxMS)
			}
			if g.Failed > 0 {
				fmt.Fprintf(tw, "%s\tfailed\t%d\t\t\t\n", label, g.Failed)
			}
			if g.FailedEvents > 0 {
				fmt.Fprintf(tw, "%s\tfailed_events\t%d\t\t\t\n", label, g.FailedEvents)
			}
		}
		group("first", b.First)
		group("wake", b.Wakes)
		tw.Flush()
	}
	if len(rep.Slowest) > 0 {
		fmt.Fprintln(w)
		fmt.Fprintln(w, "slowest first commands and wakes:")
		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "event\tenv\tinstance\tstep\tms\tattempts")
		for _, e := range rep.Slowest {
			fmt.Fprintf(tw, "%s\t%s\t%s\t%d\t%.0f\t%d\n", e.Event, e.Env, e.Instance, e.Step, e.DurMS, e.Attempts)
		}
		tw.Flush()
	}
	if len(rep.Unresolved) > 0 {
		fmt.Fprintf(w, "\nunresolved: %d instances (report -unresolved FILE writes them in exclude-file format)\n", len(rep.Unresolved))
	}
	if len(rep.Errors) > 0 {
		fmt.Fprintln(w)
		fmt.Fprintln(w, "errors:")
		for i, e := range rep.Errors {
			if i == 10 {
				fmt.Fprintf(w, "  ... %d more\n", len(rep.Errors)-10)
				break
			}
			msg := strings.ReplaceAll(e.Err, "\n", " ")
			if len(msg) > 160 {
				msg = msg[:160] + "..."
			}
			fmt.Fprintf(w, "  %5d  %s\n", e.Count, msg)
		}
	}
}

// WriteComparison prints several runs side by side, one column per run: the
// header figures, then every event in the order it first appears, then the
// per-step figures when a run has steps. It is the table for the same
// workload on different clusters or builds. names are the files the reports
// came from; a column is headed by its run id unless that is empty or shared
// with another column, in which case the file name is used.
func WriteComparison(w io.Writer, reps []*Report, names []string) {
	labels := columnLabels(reps, names)
	fmt.Fprintln(w, "cells: p50/p95 ms n=count [f=failed r=retried]; steps: live tasks [f=failed] barrier_s/park_s wake p50/p95 ms")
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	row := func(label string, cell func(*Report) string) {
		fmt.Fprint(tw, label)
		for _, r := range reps {
			fmt.Fprintf(tw, "\t%s", cell(r))
		}
		fmt.Fprintln(tw)
	}
	fmt.Fprint(tw, "run")
	for _, l := range labels {
		fmt.Fprintf(tw, "\t%s", l)
	}
	fmt.Fprintln(tw)
	row("mode", func(r *Report) string { return r.Mode })
	row("reps", func(r *Report) string { return fmt.Sprintf("%d", r.Reps) })
	row("tasks", func(r *Report) string { return fmt.Sprintf("%d", r.Tasks) })
	row("failed", func(r *Report) string { return fmt.Sprintf("%d", r.Failed) })
	row("reward_rate", func(r *Report) string { return fmt.Sprintf("%.3f", r.RewardRate) })
	row("run_s", func(r *Report) string { return fmt.Sprintf("%.1f", r.RunMS/1000) })
	builtin := map[string]bool{"run": true, "mode": true, "reps": true, "tasks": true, "failed": true, "reward_rate": true, "run_s": true}
	for _, k := range labelKeys(reps) {
		if k == "mode" && labelMatchesMode(reps) {
			continue // the loader's own label; the mode row above says it
		}
		name := k
		if builtin[k] {
			name = "label." + k // an operator label named like a built-in row
		}
		row(name, func(r *Report) string {
			if v, ok := r.Labels[k]; ok {
				return v
			}
			return "-"
		})
	}

	type key struct{ event, name string }
	var order []key
	seen := map[key]bool{}
	for _, r := range reps {
		for _, s := range r.Stats {
			k := key{s.Event, s.Name}
			if !seen[k] {
				seen[k] = true
				order = append(order, k)
			}
		}
	}
	for _, k := range order {
		label := k.event
		if k.name != "" {
			label += " " + k.name
		}
		row(label, func(r *Report) string {
			for _, s := range r.Stats {
				if s.Event != k.event || s.Name != k.name {
					continue
				}
				cell := fmt.Sprintf("%.0f/%.0f n=%d", s.P50MS, s.P95MS, s.Count)
				if s.Count == s.Failed {
					// No call succeeded: there is no latency to show.
					cell = fmt.Sprintf("-/- n=%d", s.Count)
				}
				if s.Failed > 0 {
					cell += fmt.Sprintf(" f=%d", s.Failed)
				}
				if s.Retried > 0 {
					cell += fmt.Sprintf(" r=%d", s.Retried)
				}
				return cell
			}
			return "-"
		})
	}

	last := 0
	for _, r := range reps {
		for _, s := range r.Steps {
			if s.Step > last {
				last = s.Step
			}
		}
	}
	for n := 1; n <= last; n++ {
		row(fmt.Sprintf("step %d", n), func(r *Report) string {
			for _, s := range r.Steps {
				if s.Step != n {
					continue
				}
				cell := fmt.Sprintf("live=%d tasks=%d", s.Live, s.Tasks)
				if s.Failed > 0 {
					cell += fmt.Sprintf(" f=%d", s.Failed)
				}
				wake := fmt.Sprintf("%.0f/%.0f", s.WakeP50MS, s.WakeP95MS)
				if s.WakeP50MS == 0 && s.WakeP95MS == 0 {
					wake = "-/-" // no wake in this step: the first one, or -park none
				}
				return cell + fmt.Sprintf(" %.1f/%.1f %s", s.BarrierMS/1000, s.ParkMS/1000, wake)
			}
			return "-"
		})
	}
	tw.Flush()
}

// columnLabels heads each column with its run id, or with the file name when
// the run id is empty or the same as another column's; when the file names
// also repeat, with the full path.
func columnLabels(reps []*Report, names []string) []string {
	count := map[string]int{}
	for _, r := range reps {
		count[r.RunID]++
	}
	labels := make([]string, len(reps))
	byFile := map[int]bool{}
	for i, r := range reps {
		labels[i] = r.RunID
		if r.RunID != "" && count[r.RunID] == 1 {
			continue
		}
		if i < len(names) && names[i] != "" {
			labels[i] = strings.TrimSuffix(filepath.Base(names[i]), ".jsonl")
			byFile[i] = true
		} else {
			labels[i] = fmt.Sprintf("run %d", i+1)
		}
	}
	seen := map[string]int{}
	for _, l := range labels {
		seen[l]++
	}
	for i := range labels {
		if byFile[i] && seen[labels[i]] > 1 {
			labels[i] = names[i]
		}
	}
	return labels
}

// NodeCacheSample is one reading of the image-cache counters of every node,
// as sample-node-cache.sh writes it: a node name, a tab, then a Prometheus
// text line, with one "# sampled <unix seconds>" line per node. Nodes maps
// each node the sample covered to the start time of its atelet process (zero
// when that series was absent); Requests holds ate_imagecache_requests_total
// by node and outcome.
type NodeCacheSample struct {
	// Time is when the sample was taken: the latest "# sampled" stamp.
	Time     time.Time
	Nodes    map[string]time.Time
	Requests map[string]map[string]float64
}

const (
	cacheRequestsSeries = "ate_imagecache_requests_total{"
	processStartSeries  = "process_start_time_seconds"
)

// ParseNodeCacheSample reads a sample file. Comment lines and other metric
// families only register the node; a line without the node tab, a request
// line without labels, or a value that is not a number is an error.
func ParseNodeCacheSample(r io.Reader) (*NodeCacheSample, error) {
	s := &NodeCacheSample{Nodes: map[string]time.Time{}, Requests: map[string]map[string]float64{}}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	line := 0
	for sc.Scan() {
		line++
		text := strings.TrimRight(sc.Text(), " \r")
		if text == "" {
			continue
		}
		node, rest, ok := strings.Cut(text, "\t")
		if !ok || node == "" {
			return nil, fmt.Errorf("sample line %d: no node name and tab", line)
		}
		if _, seen := s.Nodes[node]; !seen {
			s.Nodes[node] = time.Time{}
		}
		switch {
		case strings.HasPrefix(rest, "# sampled "):
			v, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimPrefix(rest, "# sampled ")), 64)
			if err != nil {
				return nil, fmt.Errorf("sample line %d: sample time %q: %w", line, rest, err)
			}
			if t := unixTime(v); t.After(s.Time) {
				s.Time = t
			}
		case strings.HasPrefix(rest, "#"):
		case strings.HasPrefix(rest, processStartSeries+" ") || strings.HasPrefix(rest, processStartSeries+"{"):
			tail := rest[len(processStartSeries):]
			if rb := strings.IndexByte(tail, '}'); strings.HasPrefix(tail, "{") && rb >= 0 {
				tail = tail[rb+1:]
			}
			v, err := lastNumber(tail)
			if err != nil {
				return nil, fmt.Errorf("sample line %d: %w", line, err)
			}
			s.Nodes[node] = unixTime(v)
		case strings.HasPrefix(rest, cacheRequestsSeries):
			rb := strings.LastIndexByte(rest, '}')
			if rb < 0 {
				return nil, fmt.Errorf("sample line %d: no labels", line)
			}
			outcome := ""
			for _, kv := range strings.Split(rest[len(cacheRequestsSeries):rb], ",") {
				if k, v, ok := strings.Cut(strings.TrimSpace(kv), "="); ok && k == "ate_imagecache_outcome" {
					outcome = strings.Trim(v, `"`)
				}
			}
			if outcome == "" {
				return nil, fmt.Errorf("sample line %d: no ate_imagecache_outcome label", line)
			}
			v, err := lastNumber(rest[rb+1:])
			if err != nil {
				return nil, fmt.Errorf("sample line %d: %w", line, err)
			}
			if s.Requests[node] == nil {
				s.Requests[node] = map[string]float64{}
			}
			s.Requests[node][outcome] += v
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return s, nil
}

// lastNumber parses the sample value of a Prometheus text line: the first
// field after the metric name and labels (a timestamp may follow it).
func lastNumber(tail string) (float64, error) {
	fields := strings.Fields(tail)
	if len(fields) == 0 {
		return 0, errors.New("no value")
	}
	v, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return 0, fmt.Errorf("value %q: %w", fields[0], err)
	}
	return v, nil
}

func unixTime(seconds float64) time.Time {
	sec := math.Floor(seconds)
	return time.Unix(int64(sec), int64((seconds-sec)*1e9))
}

// CacheDelta is what the node image caches did between two samples. The
// misses are the cold pulls of the window: every image lookup that had to
// pull and unpack, golden bakes and wakes on a node without the image
// included.
type CacheDelta struct {
	// Nodes were in both samples with the same atelet process; their
	// counters are differenced. Restarted nodes were in both samples but
	// atelet started in between, and Added nodes were only in the after
	// sample with an atelet younger than the before sample (a node the
	// autoscaler brought in); both count from zero. Removed nodes were only in
	// the before sample. Skipped nodes were only in the after sample with an
	// atelet older than the before sample, so the before scrape failed; they
	// are reported and left out.
	Nodes     int     `json:"nodes"`
	Restarted int     `json:"restarted,omitempty"`
	Added     int     `json:"added,omitempty"`
	Removed   int     `json:"removed,omitempty"`
	Skipped   int     `json:"skipped,omitempty"`
	Hits      float64 `json:"hits"`
	Misses    float64 `json:"misses"`
	// Errors sums the error, cancelled and timeout outcomes.
	Errors          float64 `json:"errors"`
	NodesWithMisses int     `json:"nodes_with_misses"`
}

// Delta compares s, taken before a run, with after.
func (s *NodeCacheSample) Delta(after *NodeCacheSample) CacheDelta {
	var d CacheDelta
	add := func(counts map[string]float64) {
		for o, v := range counts {
			switch o {
			case "hit":
				d.Hits += v
			case "miss":
				d.Misses += v
				if v > 0 {
					d.NodesWithMisses++
				}
			default:
				d.Errors += v
			}
		}
	}
	for node, startAfter := range after.Nodes {
		startBefore, inBefore := s.Nodes[node]
		switch {
		case !inBefore:
			if !startAfter.IsZero() && !s.Time.IsZero() && startAfter.After(s.Time) {
				d.Added++
				add(after.Requests[node])
			} else {
				d.Skipped++
			}
		case !startAfter.IsZero() && !startBefore.IsZero() && !startAfter.Equal(startBefore):
			d.Restarted++
			add(after.Requests[node])
		default:
			d.Nodes++
			diff := map[string]float64{}
			for o, v := range after.Requests[node] {
				delta := v - s.Requests[node][o]
				if delta < 0 {
					// No start time to prove a restart, but the counter went down.
					delta = v
				}
				diff[o] = delta
			}
			add(diff)
		}
	}
	for node := range s.Nodes {
		if _, inAfter := after.Nodes[node]; !inAfter {
			d.Removed++
		}
	}
	return d
}

// RestoreTiming is one "Restore timing breakdown" record from atelet's log,
// as harvest-restore-timings.sh collects them: the actor restored, which is
// the environment's name, and the seconds each phase took.
type RestoreTiming struct {
	// Time is when atelet wrote the record, at the end of the restore.
	Time     time.Time
	Actor    string
	Fidelity string
	// Err is the record's error.type; empty on success.
	Err string
	// Phases maps a phase (volume_mount, manifest_fetch, sandbox_assets,
	// download, oci_unpack, worker_restore, total) to seconds.
	Phases map[string]float64
}

const restoreDurationPrefix = "ate.actor.restore.duration."

// restorePhaseOrder is the order atelet times the phases in. A phase not in
// it (an older or newer atelet) is reported after these, in name order.
var restorePhaseOrder = []string{"volume_mount", "manifest_fetch", "sandbox_assets", "download", "oci_unpack", "worker_restore", "total"}

// ParseRestoreTimings reads atelet log lines, one JSON record each, keeps
// atelet's restore records and groups them by actor in time order. Other
// records are skipped, among them the worker's own restore record, which has
// the same message but no ate.actor.restore.duration.total. A line that is
// not JSON, or a restore record without an actor name or a parseable time,
// is an error.
func ParseRestoreTimings(r io.Reader) (map[string][]RestoreTiming, error) {
	out := map[string][]RestoreTiming{}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	line := 0
	for sc.Scan() {
		line++
		text := bytes.TrimSpace(sc.Bytes())
		if len(text) == 0 {
			continue
		}
		var rec map[string]any
		if err := json.Unmarshal(text, &rec); err != nil {
			return nil, fmt.Errorf("restore log line %d: %w", line, err)
		}
		if rec["msg"] != "Restore timing breakdown" {
			continue
		}
		rt := RestoreTiming{Phases: map[string]float64{}}
		for k, v := range rec {
			if f, ok := v.(float64); ok && strings.HasPrefix(k, restoreDurationPrefix) {
				rt.Phases[strings.TrimPrefix(k, restoreDurationPrefix)] = f
			}
		}
		if _, ok := rt.Phases["total"]; !ok {
			continue
		}
		rt.Actor, _ = rec["ate.actor.name"].(string)
		if rt.Actor == "" {
			return nil, fmt.Errorf("restore log line %d: no ate.actor.name", line)
		}
		ts, ok := rec["time"].(string)
		if !ok {
			return nil, fmt.Errorf("restore log line %d: no time", line)
		}
		var err error
		if rt.Time, err = time.Parse(time.RFC3339Nano, ts); err != nil {
			return nil, fmt.Errorf("restore log line %d: time %q: %w", line, ts, err)
		}
		rt.Fidelity, _ = rec["ate.snapshot.fidelity"].(string)
		rt.Err, _ = rec["error.type"].(string)
		out[rt.Actor] = append(out[rt.Actor], rt)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	for _, rs := range out {
		sort.SliceStable(rs, func(i, j int) bool { return rs[i].Time.Before(rs[j].Time) })
	}
	return out, nil
}

// PhaseStat is one restore phase over the environments of a run.
type PhaseStat struct {
	Phase string  `json:"phase"`
	Count int     `json:"count"`
	P50MS float64 `json:"p50_ms"`
	P95MS float64 `json:"p95_ms"`
	MaxMS float64 `json:"max_ms"`
}

// RestoreGroup is the phase breakdown of the restores paired with one kind
// of event: ttfi (the environment's first start) or wake.
type RestoreGroup struct {
	Restores int `json:"restores"`
	// Failed counts records with an error.type; their phases are left out.
	Failed int `json:"failed,omitempty"`
	// FailedEvents counts restores whose event itself failed (a retried or
	// timed-out first command); their span is left out of Outside.
	FailedEvents int         `json:"failed_events,omitempty"`
	Phases       []PhaseStat `json:"phases"`
	// Outside is the event's span minus atelet's total: ateapi, the
	// scheduler, the router and the command itself. Nil when nothing paired.
	Outside *PhaseStat `json:"outside,omitempty"`
}

// RestoreBreakdown is atelet's restore records joined to a run's events.
type RestoreBreakdown struct {
	// OutsideRun counts records before the run's first event started or after
	// its last ended: an earlier run with the same run id, say.
	OutsideRun int `json:"outside_run,omitempty"`
	// Unmatched counts records for actors the run has no event for.
	Unmatched int `json:"unmatched,omitempty"`
	// Unpaired counts records inside the run whose time no ttfi or wake
	// window of their environment contains: a restore after an eviction, or
	// after a failed suspend.
	Unpaired int          `json:"unpaired,omitempty"`
	First    RestoreGroup `json:"first"`
	Wakes    RestoreGroup `json:"wakes"`
}

// pairSlack absorbs the clock difference between the loader and a node.
const pairSlack = 2 * time.Second

// JoinRestores pairs each restore record with the ttfi or wake event of its
// environment whose window, the event's end minus its duration up to its
// end, contains the record's time. A record paired with a ttfi event is a
// first start; one paired with a wake is a wake. Records outside the run's
// span, for unknown actors, or inside no window are counted, not paired.
func JoinRestores(events []Event, timings map[string][]RestoreTiming) *RestoreBreakdown {
	b := &RestoreBreakdown{}
	windows := map[string][]Event{}
	var spanStart, spanEnd time.Time
	for _, e := range events {
		if e.Event != EventTTFI && e.Event != EventWake {
			continue
		}
		windows[e.Env] = append(windows[e.Env], e)
		if start := e.Time.Add(-time.Duration(e.DurMS * float64(time.Millisecond))); spanStart.IsZero() || start.Before(spanStart) {
			spanStart = start
		}
		if spanEnd.IsZero() || e.Time.After(spanEnd) {
			spanEnd = e.Time
		}
	}
	var firstRecs, wakeRecs []RestoreTiming
	var firstOutside, wakeOutside []float64
	for actor, recs := range timings {
		for _, rt := range recs {
			if spanStart.IsZero() || rt.Time.Before(spanStart.Add(-pairSlack)) || rt.Time.After(spanEnd.Add(pairSlack)) {
				b.OutsideRun++
				continue
			}
			evs := windows[actor]
			if len(evs) == 0 {
				b.Unmatched++
				continue
			}
			var ev *Event
			for i := range evs {
				start := evs[i].Time.Add(-time.Duration(evs[i].DurMS * float64(time.Millisecond)))
				if rt.Time.Before(start.Add(-pairSlack)) || rt.Time.After(evs[i].Time.Add(pairSlack)) {
					continue
				}
				if ev == nil || evs[i].Time.Before(ev.Time) {
					ev = &evs[i]
				}
			}
			if ev == nil {
				b.Unpaired++
				continue
			}
			g, recsOut, outside := &b.Wakes, &wakeRecs, &wakeOutside
			if ev.Event == EventTTFI {
				g, recsOut, outside = &b.First, &firstRecs, &firstOutside
			}
			if rt.Err != "" {
				g.Failed++
				continue
			}
			*recsOut = append(*recsOut, rt)
			if !ev.OK {
				g.FailedEvents++
				continue
			}
			*outside = append(*outside, ev.DurMS-rt.Phases["total"]*1000)
		}
	}
	b.First.Restores, b.First.Phases, b.First.Outside = len(firstRecs), phaseStats(firstRecs), spanStat("outside", firstOutside)
	b.Wakes.Restores, b.Wakes.Phases, b.Wakes.Outside = len(wakeRecs), phaseStats(wakeRecs), spanStat("outside", wakeOutside)
	return b
}

// phaseStats gives the known phases in atelet's order, then any other phase
// the records carry in name order, so an atelet that names a phase
// differently still shows up.
func phaseStats(recs []RestoreTiming) []PhaseStat {
	known := map[string]bool{}
	for _, p := range restorePhaseOrder {
		known[p] = true
	}
	var extra []string
	seen := map[string]bool{}
	for _, rt := range recs {
		for p := range rt.Phases {
			if !known[p] && !seen[p] {
				seen[p] = true
				extra = append(extra, p)
			}
		}
	}
	sort.Strings(extra)
	var out []PhaseStat
	for _, phase := range append(append([]string{}, restorePhaseOrder...), extra...) {
		var d []float64
		for _, rt := range recs {
			if v, ok := rt.Phases[phase]; ok {
				d = append(d, v*1000)
			}
		}
		if len(d) == 0 {
			continue
		}
		sort.Float64s(d)
		out = append(out, PhaseStat{Phase: phase, Count: len(d), P50MS: percentile(d, 0.50), P95MS: percentile(d, 0.95), MaxMS: percentile(d, 1)})
	}
	return out
}

// spanStat summarizes event spans under name; nil when there are none.
func spanStat(name string, d []float64) *PhaseStat {
	if len(d) == 0 {
		return nil
	}
	sort.Float64s(d)
	return &PhaseStat{Phase: name, Count: len(d), P50MS: percentile(d, 0.50), P95MS: percentile(d, 0.95), MaxMS: percentile(d, 1)}
}

// labelMatchesMode reports whether every report's mode label, where it has
// one, is its mode: the loader's own label, which the mode row already shows.
func labelMatchesMode(reps []*Report) bool {
	for _, r := range reps {
		if v, ok := r.Labels["mode"]; ok && v != r.Mode {
			return false
		}
	}
	return true
}

// labelKeys is the sorted union of the reports' label keys.
func labelKeys(reps []*Report) []string {
	seen := map[string]bool{}
	for _, r := range reps {
		for k := range r.Labels {
			seen[k] = true
		}
	}
	keys := make([]string, 0, len(seen))
	for k := range seen {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// labelLine renders labels as "k=v k=v", keys sorted.
func labelLine(labels map[string]string) string {
	parts := make([]string, 0, len(labels))
	for _, k := range labelKeys([]*Report{{Labels: labels}}) {
		parts = append(parts, k+"="+labels[k])
	}
	return strings.Join(parts, " ")
}
