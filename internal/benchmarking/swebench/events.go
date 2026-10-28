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
	"encoding/json"
	"io"
	"sync"
	"time"
)

// Event names. Every timed operation of a run is one of these.
const (
	EventCreate  = "create"       // CreateEnvironment call
	EventTTFI    = "ttfi"         // create start to the first command's return
	EventWake    = "wake"         // first command after a park, including the wake it triggers
	EventStep    = "step"         // one trajectory step (name is the step name)
	EventTask    = "task"         // one trajectory replay end to end
	EventSuspend = "suspend"      // SuspendEnvironment call
	EventDelete  = "delete"       // DeleteEnvironment call
	EventBarrier = "step_barrier" // steps mode: first task start to last task end in a training step
	EventPark    = "park"         // steps mode: parking the whole fleet between steps
	EventRun     = "run"          // whole run
)

// Event is one JSONL record. Dur is wall time as seen by the loader; it
// includes the network and the API server, which is what a trainer sees.
type Event struct {
	Time  time.Time `json:"ts"`
	RunID string    `json:"run"`
	Event string    `json:"event"`
	Name  string    `json:"name,omitempty"`
	// Env is the environment's name. With a delete park, an environment is
	// recreated under "<name>-s<step>" from step 2 on, thus group a replay's
	// events by Instance and Task, not by Env.
	Env      string `json:"env,omitempty"`
	Instance string `json:"instance,omitempty"`
	// Step is the training step in steps mode (1-based); 0 in eval mode.
	Step int `json:"step,omitempty"`
	// Task numbers the replays of one instance (eval: task index; steps:
	// rollout index).
	Task   int     `json:"task,omitempty"`
	DurMS  float64 `json:"dur_ms"`
	OK     bool    `json:"ok"`
	Exit   *int    `json:"exit,omitempty"`
	Reward *int    `json:"reward,omitempty"`
	// Detail is the dataset grader's short verdict string for a reward step
	// (for example "f2p 1/1 p2p 3/4" or "invalid run").
	Detail string `json:"detail,omitempty"`
	Err    string `json:"err,omitempty"`
	// Count is the number of tasks or environments an aggregate event covers.
	Count int `json:"count,omitempty"`
	// Tail is the last bytes of a failed or reward step's output.
	Tail string `json:"tail,omitempty"`
	// Attempts is how many tries a ttfi or wake took; above 1 means the
	// environment was not serving when first asked.
	Attempts int `json:"attempts,omitempty"`
	// Labels describe the run, on its run event: the loader's own (mode,
	// width, images, trajectory, the shape flags) and the operator's -label
	// values, which win on a clash.
	Labels map[string]string `json:"labels,omitempty"`

	// Working-set telemetry attached to step_barrier, park, and run events when
	// a scheduled working-set pool is active.
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

const recorderBufferSize = 64 * 1024

// Recorder writes events as buffered JSON lines. It is safe for concurrent use.
type Recorder struct {
	mu  sync.Mutex
	bw  *bufio.Writer
	enc *json.Encoder
	n   int
	err error
}

// NewRecorder writes to w through a 64 KiB buffer; a nil w discards events.
func NewRecorder(w io.Writer) *Recorder {
	if w == nil {
		return &Recorder{enc: json.NewEncoder(io.Discard)}
	}
	bw := bufio.NewWriterSize(w, recorderBufferSize)
	return &Recorder{bw: bw, enc: json.NewEncoder(bw)}
}

// Record writes one event, stamping the time if unset. Coarse boundary events
// (step_barrier, park, run) flush the underlying buffer automatically.
func (r *Recorder) Record(e Event) {
	if e.Time.IsZero() {
		e.Time = time.Now()
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if err := r.enc.Encode(e); err != nil && r.err == nil {
		r.err = err
	}
	if r.bw != nil && (e.Event == EventBarrier || e.Event == EventPark || e.Event == EventRun) {
		if err := r.bw.Flush(); err != nil && r.err == nil {
			r.err = err
		}
	}
	r.n++
}

// Flush flushes any buffered events to the underlying writer and returns the
// first write or flush error.
func (r *Recorder) Flush() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.flushLocked()
	return r.err
}

// Close flushes buffered events and returns the first write or flush error.
func (r *Recorder) Close() error {
	return r.Flush()
}

func (r *Recorder) flushLocked() {
	if r.bw != nil {
		if err := r.bw.Flush(); err != nil && r.err == nil {
			r.err = err
		}
	}
}

// Len is the number of events written so far.
func (r *Recorder) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.n
}

// Err flushes any buffered events and returns the first write error, if any;
// a run whose events were lost must not report success.
func (r *Recorder) Err() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.flushLocked()
	return r.err
}

func intp(v int) *int { return &v }
