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
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/agent-substrate/env/internal/benchmarking/schema"
)

// ForkableBackend is an optional extension to Backend for backends that support
// creating a child sandbox from a suspended golden base sandbox.
type ForkableBackend interface {
	Backend
	Fork(ctx context.Context, parentID, childID, image string) (Sandbox, error)
}

// WorkingSetEntry represents a sandbox managed by a WorkingSet.
type WorkingSetEntry struct {
	SandboxID   string
	BaseID      string
	Image       string
	InstanceID  string
	Sandbox     Sandbox
	CreateStart time.Time
	CreatedStep int
	LastStep    int
	ReuseCount  int
	Hit         bool
	WasParked   bool
	IsForkChild bool

	asyncDone chan struct{}
	asyncDur  time.Duration
	asyncEnd  time.Time
	asyncErr  error
}

// WorkingSetStats summarizes per-step or cumulative working-set behavior.
type WorkingSetStats struct {
	Hits                 int
	Misses               int
	HitRate              float64
	StandingPoolSize     int
	Created              int
	Reused               int
	Retired              int
	PrefetchLeadMarginMS float64
	ReleaseOverlapMS     float64
}

// WorkingSetConfig configures a WorkingSet manager for a job.
type WorkingSetConfig struct {
	JobID      string
	Backend    Backend
	ParkPolicy schema.ParkPolicy
	// MaxParked caps the number of idle sandboxes retained in the standing pool.
	// Zero means unbounded.
	MaxParked int
}

type prewarmFuture struct {
	done   chan struct{}
	doneAt time.Time
	err    error
}

// WorkingSet manages a client-side standing pool of parked sandboxes across
// training steps, matching each step's scheduled tasks by Image.
type WorkingSet struct {
	cfg     WorkingSetConfig
	backend Backend

	mu          sync.Mutex
	parked      map[string][]*WorkingSetEntry // image -> FIFO queue of parked sandboxes
	parkedOrder []*WorkingSetEntry            // LRU order (oldest first) across all images
	forkBases   map[string]*WorkingSetEntry   // image -> golden base sandbox for ParkFork
	inFlight    []*WorkingSetEntry            // in-flight async suspends
	prewarms    map[int]*prewarmFuture        // step -> async prewarm job

	seq       atomic.Uint64
	totalHits int
	totalMiss int
	created   int
	reused    int
	retired   int

	stepHits    int
	stepMisses  int
	stepCreated int
	stepReused  int
	stepRetired int
}

// NewWorkingSet constructs a WorkingSet for the given configuration.
func NewWorkingSet(cfg WorkingSetConfig) (*WorkingSet, error) {
	if cfg.JobID == "" {
		return nil, errors.New("swebench: workingset job_id is required")
	}
	if cfg.Backend == nil {
		return nil, errors.New("swebench: workingset backend is required")
	}
	norm, err := schema.NormalizeParkPolicy(cfg.ParkPolicy)
	if err != nil {
		return nil, err
	}
	cfg.ParkPolicy = norm
	return &WorkingSet{
		cfg:       cfg,
		backend:   cfg.Backend,
		parked:    make(map[string][]*WorkingSetEntry),
		forkBases: make(map[string]*WorkingSetEntry),
		prewarms:  make(map[int]*prewarmFuture),
	}, nil
}

// ParkPolicy returns the normalized park policy of this WorkingSet.
func (ws *WorkingSet) ParkPolicy() schema.ParkPolicy {
	return ws.cfg.ParkPolicy
}

func (ws *WorkingSet) nextSandboxID(instanceID string, step, taskIdx int) string {
	n := ws.seq.Add(1)
	base := EnvID(ws.cfg.JobID, instanceID, taskIdx)
	suffix := fmt.Sprintf("-s%d-%d", step, n)
	if len(base)+len(suffix) > maxNameLen {
		base = base[:maxNameLen-len(suffix)]
	}
	return base + suffix
}

// BeginStep resets the per-step working-set counters at the start of a step.
func (ws *WorkingSet) BeginStep() {
	ws.mu.Lock()
	defer ws.mu.Unlock()
	ws.stepHits = 0
	ws.stepMisses = 0
	ws.stepCreated = 0
	ws.stepReused = 0
	ws.stepRetired = 0
}

// Acquire obtains a sandbox for (instanceID, image) at the given step and taskIndex.
// If a parked sandbox with the same image is available in the pool (or a golden
// base snapshot exists under ParkFork), it is reused (Hit = true); otherwise a
// new sandbox is created (Hit = false).
func (ws *WorkingSet) Acquire(ctx context.Context, step, taskIndex int, instanceID, image string) (*WorkingSetEntry, error) {
	if ws.cfg.ParkPolicy == schema.ParkFork {
		return ws.acquireFork(ctx, step, taskIndex, instanceID, image)
	}

	ws.mu.Lock()
	queue := ws.parked[image]
	var hit *WorkingSetEntry
	if len(queue) > 0 {
		hit = queue[0]
		ws.parked[image] = queue[1:]
		ws.removeFromOrderLocked(hit)
		ws.totalHits++
		ws.reused++
		ws.stepHits++
		ws.stepReused++
	} else {
		ws.totalMiss++
		ws.created++
		ws.stepMisses++
		ws.stepCreated++
	}
	ws.mu.Unlock()

	if hit != nil {
		if hit.asyncDone != nil {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-hit.asyncDone:
			}
			if hit.asyncErr != nil {
				return nil, hit.asyncErr
			}
		}
		hit.InstanceID = instanceID
		hit.LastStep = step
		hit.ReuseCount++
		hit.Hit = true
		return hit, nil
	}

	sbID := ws.nextSandboxID(instanceID, step, taskIndex)
	createStart := time.Now()
	sb, err := ws.backend.Create(ctx, sbID, image)
	if err != nil {
		return nil, err
	}
	return &WorkingSetEntry{
		SandboxID:   sbID,
		BaseID:      sbID,
		Image:       image,
		InstanceID:  instanceID,
		Sandbox:     sb,
		CreateStart: createStart,
		CreatedStep: step,
		LastStep:    step,
		ReuseCount:  0,
		Hit:         false,
		WasParked:   false,
	}, nil
}

func (ws *WorkingSet) acquireFork(ctx context.Context, step, taskIndex int, instanceID, image string) (*WorkingSetEntry, error) {
	ws.mu.Lock()
	base, hasBase := ws.forkBases[image]
	if hasBase {
		ws.totalHits++
		ws.reused++
		ws.stepHits++
		ws.stepReused++
	} else {
		ws.totalMiss++
		ws.created++
		ws.stepMisses++
		ws.stepCreated++
	}
	ws.mu.Unlock()

	if !hasBase {
		baseID := ws.nextSandboxID("base-"+instanceID, step, taskIndex)
		createStart := time.Now()
		baseSB, err := ws.backend.Create(ctx, baseID, image)
		if err != nil {
			return nil, err
		}
		if err := baseSB.Suspend(ctx); err != nil {
			_ = baseSB.Delete(context.Background())
			return nil, err
		}
		base = &WorkingSetEntry{
			SandboxID:   baseID,
			BaseID:      baseID,
			Image:       image,
			InstanceID:  instanceID,
			Sandbox:     baseSB,
			CreateStart: createStart,
			CreatedStep: step,
			LastStep:    step,
			WasParked:   true,
		}
		ws.mu.Lock()
		if existing, ok := ws.forkBases[image]; ok {
			ws.mu.Unlock()
			_ = baseSB.Delete(context.Background())
			base = existing
		} else {
			ws.forkBases[image] = base
			ws.mu.Unlock()
		}
	}

	childID := ws.nextSandboxID(instanceID, step, taskIndex)
	createStart := time.Now()
	var (
		childSB Sandbox
		err     error
	)
	if fb, ok := ws.backend.(ForkableBackend); ok {
		childSB, err = fb.Fork(ctx, base.SandboxID, childID, image)
	} else {
		childSB, err = ws.backend.Create(ctx, childID, image)
	}
	if err != nil {
		return nil, err
	}
	return &WorkingSetEntry{
		SandboxID:   childID,
		BaseID:      base.SandboxID,
		Image:       image,
		InstanceID:  instanceID,
		Sandbox:     childSB,
		CreateStart: createStart,
		CreatedStep: step,
		LastStep:    step,
		ReuseCount:  base.ReuseCount + 1,
		Hit:         hasBase,
		WasParked:   hasBase,
		IsForkChild: true,
	}, nil
}

// Release parks or retires an active sandbox at the end of its task or step
// according to the WorkingSet's ParkPolicy.
func (ws *WorkingSet) Release(ctx context.Context, entry *WorkingSetEntry) error {
	if entry == nil || entry.Sandbox == nil {
		return nil
	}
	switch ws.cfg.ParkPolicy {
	case schema.ParkDelete:
		err := entry.Sandbox.Delete(ctx)
		ws.mu.Lock()
		ws.retired++
		ws.stepRetired++
		ws.mu.Unlock()
		return err

	case schema.ParkFork:
		if entry.IsForkChild {
			err := entry.Sandbox.Delete(ctx)
			ws.mu.Lock()
			ws.retired++
			ws.stepRetired++
			ws.mu.Unlock()
			return err
		}
		return nil

	case schema.ParkNone:
		entry.WasParked = false
		entry.asyncDone = nil
		return ws.enqueueParked(ctx, entry)

	case schema.ParkSuspendSync:
		start := time.Now()
		if err := entry.Sandbox.Suspend(ctx); err != nil {
			_ = entry.Sandbox.Delete(context.Background())
			ws.mu.Lock()
			ws.retired++
			ws.stepRetired++
			ws.mu.Unlock()
			return err
		}
		entry.WasParked = true
		entry.asyncDur = time.Since(start)
		entry.asyncEnd = time.Now()
		entry.asyncDone = nil
		return ws.enqueueParked(ctx, entry)

	case schema.ParkSuspendAsync:
		done := make(chan struct{})
		entry.asyncDone = done
		entry.asyncErr = nil
		entry.WasParked = true
		ws.mu.Lock()
		ws.inFlight = append(ws.inFlight, entry)
		ws.mu.Unlock()
		if err := ws.enqueueParked(ctx, entry); err != nil {
			close(done)
			return err
		}
		go func(e *WorkingSetEntry) {
			start := time.Now()
			err := e.Sandbox.Suspend(ctx)
			e.asyncDur = time.Since(start)
			e.asyncEnd = time.Now()
			e.asyncErr = err
			close(done)
		}(entry)
		return nil

	default:
		return fmt.Errorf("swebench: unsupported park policy %q", ws.cfg.ParkPolicy)
	}
}

// AwaitAsyncSuspends waits for all in-flight asynchronous suspends launched
// during the current step and returns the total overlapped suspend duration
// (in milliseconds) that completed before stepResumeAt.
func (ws *WorkingSet) AwaitAsyncSuspends(ctx context.Context, stepBarrierAt, stepResumeAt time.Time) (float64, error) {
	ws.mu.Lock()
	pending := ws.inFlight
	ws.inFlight = nil
	ws.mu.Unlock()

	var (
		overlapMS float64
		firstErr  error
	)
	for _, e := range pending {
		if e.asyncDone != nil {
			select {
			case <-ctx.Done():
				return overlapMS, ctx.Err()
			case <-e.asyncDone:
			}
			if e.asyncErr != nil && firstErr == nil {
				firstErr = e.asyncErr
			}
			dur := e.asyncDur
			if !stepResumeAt.IsZero() && !stepBarrierAt.IsZero() {
				pauseWindow := stepResumeAt.Sub(stepBarrierAt)
				if pauseWindow > 0 && dur < pauseWindow {
					overlapMS += ms(dur)
				} else if pauseWindow > 0 {
					overlapMS += ms(pauseWindow)
				} else {
					overlapMS += ms(dur)
				}
			} else {
				overlapMS += ms(dur)
			}
		}
	}
	return overlapMS, firstErr
}

// Prewarm asynchronously creates and parks sandboxes for nextStep's prefetchImages
// off the critical path when they are not already present in the standing pool.
func (ws *WorkingSet) Prewarm(ctx context.Context, nextStep int, prefetchImages []string) {
	if len(prefetchImages) == 0 {
		return
	}
	fut := &prewarmFuture{done: make(chan struct{})}
	ws.mu.Lock()
	ws.prewarms[nextStep] = fut
	ws.mu.Unlock()

	go func() {
		defer func() {
			fut.doneAt = time.Now()
			close(fut.done)
		}()
		for i, img := range prefetchImages {
			if ctx.Err() != nil {
				fut.err = ctx.Err()
				return
			}
			ws.mu.Lock()
			alreadyParked := len(ws.parked[img]) > 0
			if ws.cfg.ParkPolicy == schema.ParkFork && ws.forkBases[img] != nil {
				alreadyParked = true
			}
			ws.mu.Unlock()
			if alreadyParked {
				continue
			}

			sbID := ws.nextSandboxID("prewarm", nextStep, i)
			createStart := time.Now()
			sb, err := ws.backend.Create(ctx, sbID, img)
			if err != nil {
				fut.err = err
				continue
			}
			ws.mu.Lock()
			ws.created++
			ws.mu.Unlock()

			if ws.cfg.ParkPolicy == schema.ParkFork {
				if err := sb.Suspend(ctx); err != nil {
					_ = sb.Delete(context.Background())
					fut.err = err
					continue
				}
				ws.mu.Lock()
				if ws.forkBases[img] == nil {
					ws.forkBases[img] = &WorkingSetEntry{
						SandboxID:   sbID,
						BaseID:      sbID,
						Image:       img,
						Sandbox:     sb,
						CreateStart: createStart,
						CreatedStep: nextStep,
						LastStep:    nextStep - 1,
						WasParked:   true,
					}
					ws.mu.Unlock()
				} else {
					ws.mu.Unlock()
					_ = sb.Delete(context.Background())
				}
				continue
			}

			wasParked := false
			if ws.cfg.ParkPolicy == schema.ParkSuspendAsync || ws.cfg.ParkPolicy == schema.ParkSuspendSync {
				if err := sb.Suspend(ctx); err != nil {
					_ = sb.Delete(context.Background())
					fut.err = err
					continue
				}
				wasParked = true
			}
			entry := &WorkingSetEntry{
				SandboxID:   sbID,
				BaseID:      sbID,
				Image:       img,
				Sandbox:     sb,
				CreateStart: createStart,
				CreatedStep: nextStep,
				LastStep:    nextStep - 1,
				WasParked:   wasParked,
			}
			_ = ws.enqueueParked(ctx, entry)
		}
	}()
}

// WaitPrewarm waits for any async Prewarm scheduled for step to finish and
// returns the prefetch lead margin in milliseconds (positive when prewarm
// finished before stepStart).
func (ws *WorkingSet) WaitPrewarm(ctx context.Context, step int, stepStart time.Time) (float64, error) {
	ws.mu.Lock()
	fut, ok := ws.prewarms[step]
	delete(ws.prewarms, step)
	ws.mu.Unlock()
	if !ok || fut == nil {
		return 0, nil
	}
	select {
	case <-ctx.Done():
		return 0, ctx.Err()
	case <-fut.done:
	}
	lead := stepStart.Sub(fut.doneAt)
	if lead < 0 {
		lead = 0
	}
	return ms(lead), fut.err
}

func (ws *WorkingSet) enqueueParked(ctx context.Context, entry *WorkingSetEntry) error {
	ws.mu.Lock()
	var evict []*WorkingSetEntry
	if ws.cfg.MaxParked > 0 {
		for len(ws.parkedOrder)+1 > ws.cfg.MaxParked && len(ws.parkedOrder) > 0 {
			oldest := ws.parkedOrder[0]
			ws.parkedOrder = ws.parkedOrder[1:]
			ws.removeFromImageQueueLocked(oldest)
			ws.retired++
			ws.stepRetired++
			evict = append(evict, oldest)
		}
	}
	ws.parked[entry.Image] = append(ws.parked[entry.Image], entry)
	ws.parkedOrder = append(ws.parkedOrder, entry)
	ws.mu.Unlock()

	var firstErr error
	for _, old := range evict {
		if old.asyncDone != nil {
			<-old.asyncDone
		}
		if err := old.Sandbox.Delete(ctx); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// RetireSurplus deletes parked sandboxes whose images are not in keepImages
// until the standing pool size is at most maxKeep.
func (ws *WorkingSet) RetireSurplus(ctx context.Context, keepImages map[string]bool, maxKeep int) (int, error) {
	ws.mu.Lock()
	var toDelete []*WorkingSetEntry
	remaining := make([]*WorkingSetEntry, 0, len(ws.parkedOrder))
	for _, e := range ws.parkedOrder {
		if !keepImages[e.Image] && (maxKeep <= 0 || len(ws.parkedOrder)-len(toDelete) > maxKeep) {
			ws.removeFromImageQueueLocked(e)
			ws.retired++
			ws.stepRetired++
			toDelete = append(toDelete, e)
		} else {
			remaining = append(remaining, e)
		}
	}
	ws.parkedOrder = remaining
	ws.mu.Unlock()

	var firstErr error
	for _, e := range toDelete {
		if e.asyncDone != nil {
			<-e.asyncDone
		}
		if err := e.Sandbox.Delete(ctx); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return len(toDelete), firstErr
}

// Close deletes all parked and fork-base sandboxes remaining in the pool.
func (ws *WorkingSet) Close(ctx context.Context) error {
	ws.mu.Lock()
	all := append([]*WorkingSetEntry(nil), ws.parkedOrder...)
	for _, b := range ws.forkBases {
		all = append(all, b)
	}
	ws.parked = make(map[string][]*WorkingSetEntry)
	ws.parkedOrder = nil
	ws.forkBases = make(map[string]*WorkingSetEntry)
	ws.retired += len(all)
	ws.mu.Unlock()

	var firstErr error
	for _, e := range all {
		if e.asyncDone != nil {
			<-e.asyncDone
		}
		if e.Sandbox != nil {
			if err := e.Sandbox.Delete(ctx); err != nil && !isNotFound(err) && firstErr == nil {
				firstErr = err
			}
		}
	}
	return firstErr
}

// StandingPoolSize returns the current number of parked sandboxes in the pool.
func (ws *WorkingSet) StandingPoolSize() int {
	ws.mu.Lock()
	defer ws.mu.Unlock()
	if ws.cfg.ParkPolicy == schema.ParkFork {
		return len(ws.forkBases)
	}
	return len(ws.parkedOrder)
}

// StepStats returns the working-set metrics for the current step.
func (ws *WorkingSet) StepStats(prefetchLeadMS, releaseOverlapMS float64) WorkingSetStats {
	ws.mu.Lock()
	defer ws.mu.Unlock()
	total := ws.stepHits + ws.stepMisses
	var rate float64
	if total > 0 {
		rate = float64(ws.stepHits) / float64(total)
	}
	pool := len(ws.parkedOrder)
	if ws.cfg.ParkPolicy == schema.ParkFork {
		pool = len(ws.forkBases)
	}
	return WorkingSetStats{
		Hits:                 ws.stepHits,
		Misses:               ws.stepMisses,
		HitRate:              rate,
		StandingPoolSize:     pool,
		Created:              ws.stepCreated,
		Reused:               ws.stepReused,
		Retired:              ws.stepRetired,
		PrefetchLeadMarginMS: prefetchLeadMS,
		ReleaseOverlapMS:     releaseOverlapMS,
	}
}

// SummaryStats returns the cumulative working-set metrics across the entire run.
func (ws *WorkingSet) SummaryStats(prefetchLeadMS, releaseOverlapMS float64) WorkingSetStats {
	ws.mu.Lock()
	defer ws.mu.Unlock()
	total := ws.totalHits + ws.totalMiss
	var rate float64
	if total > 0 {
		rate = float64(ws.totalHits) / float64(total)
	}
	pool := len(ws.parkedOrder)
	if ws.cfg.ParkPolicy == schema.ParkFork {
		pool = len(ws.forkBases)
	}
	return WorkingSetStats{
		Hits:                 ws.totalHits,
		Misses:               ws.totalMiss,
		HitRate:              rate,
		StandingPoolSize:     pool,
		Created:              ws.created,
		Reused:               ws.reused,
		Retired:              ws.retired,
		PrefetchLeadMarginMS: prefetchLeadMS,
		ReleaseOverlapMS:     releaseOverlapMS,
	}
}

func (ws *WorkingSet) removeFromOrderLocked(target *WorkingSetEntry) {
	for i, e := range ws.parkedOrder {
		if e == target {
			ws.parkedOrder = append(ws.parkedOrder[:i], ws.parkedOrder[i+1:]...)
			return
		}
	}
}

func (ws *WorkingSet) removeFromImageQueueLocked(target *WorkingSetEntry) {
	q := ws.parked[target.Image]
	for i, e := range q {
		if e == target {
			ws.parked[target.Image] = append(q[:i], q[i+1:]...)
			if len(ws.parked[target.Image]) == 0 {
				delete(ws.parked, target.Image)
			}
			return
		}
	}
}
