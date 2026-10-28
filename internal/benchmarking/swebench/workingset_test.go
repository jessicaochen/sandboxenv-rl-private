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
	"bytes"
	"context"
	"fmt"
	"io/fs"
	"sync/atomic"
	"testing"
	"time"

	"github.com/agent-substrate/env/internal/benchmarking/schema"
)

type wsFakeBackend struct {
	creates  atomic.Int64
	suspends atomic.Int64
	deletes  atomic.Int64
	forks    atomic.Int64
	suspWait time.Duration
}

func (b *wsFakeBackend) Create(_ context.Context, id, _ string) (Sandbox, error) {
	b.creates.Add(1)
	return &wsFakeSandbox{id: id, b: b}, nil
}

func (b *wsFakeBackend) Delete(_ context.Context, _ string) error {
	b.deletes.Add(1)
	return nil
}

func (b *wsFakeBackend) Fork(_ context.Context, _, childID, _ string) (Sandbox, error) {
	b.forks.Add(1)
	return &wsFakeSandbox{id: childID, b: b}, nil
}

type wsFakeSandbox struct {
	id string
	b  *wsFakeBackend
}

func (s *wsFakeSandbox) Run(_ context.Context, _ string, _ time.Duration) (Result, error) {
	return Result{ExitCode: 0, Stdout: "ok\n"}, nil
}

func (s *wsFakeSandbox) WriteFile(_ context.Context, _ string, _ []byte, _ fs.FileMode) error {
	return nil
}

func (s *wsFakeSandbox) Suspend(_ context.Context) error {
	if s.b.suspWait > 0 {
		time.Sleep(s.b.suspWait)
	}
	s.b.suspends.Add(1)
	return nil
}

func (s *wsFakeSandbox) Delete(_ context.Context) error {
	s.b.deletes.Add(1)
	return nil
}

func TestWorkingSetHitMissAndChurnAcrossPolicies(t *testing.T) {
	ctx := context.Background()

	// 1. suspend-sync: Step 1 creates 2 (misses), Step 2 reuses 1 (hit) and creates 1 (miss).
	bSync := &wsFakeBackend{}
	ws, err := NewWorkingSet(WorkingSetConfig{
		JobID:      "job-sync",
		Backend:    bSync,
		ParkPolicy: schema.ParkSuspendSync,
		MaxParked:  2,
	})
	if err != nil {
		t.Fatal(err)
	}
	ws.BeginStep()
	e1, err := ws.Acquire(ctx, 1, 0, "inst-a", "img-a")
	if err != nil || e1.Hit {
		t.Fatalf("step 1 acquire a: hit=%v err=%v", e1.Hit, err)
	}
	e2, err := ws.Acquire(ctx, 1, 1, "inst-b", "img-b")
	if err != nil || e2.Hit {
		t.Fatalf("step 1 acquire b: hit=%v err=%v", e2.Hit, err)
	}
	if err := ws.Release(ctx, e1); err != nil {
		t.Fatal(err)
	}
	if err := ws.Release(ctx, e2); err != nil {
		t.Fatal(err)
	}
	st1 := ws.StepStats(0, 0)
	if st1.Hits != 0 || st1.Misses != 2 || st1.StandingPoolSize != 2 {
		t.Fatalf("step 1 stats = %+v", st1)
	}

	ws.BeginStep()
	e3, err := ws.Acquire(ctx, 2, 0, "inst-a", "img-a")
	if err != nil || !e3.Hit || !e3.WasParked {
		t.Fatalf("step 2 acquire a: want hit, got hit=%v wasParked=%v err=%v", e3.Hit, e3.WasParked, err)
	}
	e4, err := ws.Acquire(ctx, 2, 1, "inst-c", "img-c")
	if err != nil || e4.Hit {
		t.Fatalf("step 2 acquire c: want miss, got hit=%v err=%v", e4.Hit, err)
	}
	if err := ws.Release(ctx, e3); err != nil {
		t.Fatal(err)
	}
	if err := ws.Release(ctx, e4); err != nil {
		t.Fatal(err)
	}
	st2 := ws.StepStats(0, 0)
	if st2.Hits != 1 || st2.Misses != 1 || st2.HitRate != 0.5 || st2.StandingPoolSize != 2 || st2.Retired != 1 {
		t.Fatalf("step 2 stats = %+v", st2)
	}
	if err := ws.Close(ctx); err != nil {
		t.Fatal(err)
	}

	// 2. delete policy: 0 hits on step 2, standing pool stays 0.
	bDel := &wsFakeBackend{}
	wsDel, err := NewWorkingSet(WorkingSetConfig{
		JobID:      "job-del",
		Backend:    bDel,
		ParkPolicy: schema.ParkDelete,
	})
	if err != nil {
		t.Fatal(err)
	}
	d1, _ := wsDel.Acquire(ctx, 1, 0, "inst-a", "img-a")
	_ = wsDel.Release(ctx, d1)
	if wsDel.StandingPoolSize() != 0 {
		t.Fatalf("delete standing pool = %d, want 0", wsDel.StandingPoolSize())
	}
	d2, _ := wsDel.Acquire(ctx, 2, 0, "inst-a", "img-a")
	if d2.Hit {
		t.Fatal("delete policy must not produce working-set hits")
	}
	_ = wsDel.Release(ctx, d2)

	// 3. fork policy: step 1 creates golden base + forks child; step 2 forks from golden base (hit).
	bFork := &wsFakeBackend{}
	wsFork, err := NewWorkingSet(WorkingSetConfig{
		JobID:      "job-fork",
		Backend:    bFork,
		ParkPolicy: schema.ParkFork,
	})
	if err != nil {
		t.Fatal(err)
	}
	f1, err := wsFork.Acquire(ctx, 1, 0, "inst-a", "img-a")
	if err != nil || f1.Hit || !f1.IsForkChild {
		t.Fatalf("fork step 1 = %+v, %v", f1, err)
	}
	_ = wsFork.Release(ctx, f1)
	f2, err := wsFork.Acquire(ctx, 2, 0, "inst-a", "img-a")
	if err != nil || !f2.Hit || !f2.IsForkChild {
		t.Fatalf("fork step 2 = %+v, %v", f2, err)
	}
	_ = wsFork.Release(ctx, f2)
	if bFork.creates.Load() != 1 || bFork.forks.Load() != 2 {
		t.Fatalf("fork backend creates=%d forks=%d, want 1 create and 2 forks", bFork.creates.Load(), bFork.forks.Load())
	}
	_ = wsFork.Close(ctx)
}

func TestWorkingSetPrewarmAndSuspendAsyncOverlap(t *testing.T) {
	ctx := context.Background()
	b := &wsFakeBackend{suspWait: 5 * time.Millisecond}
	ws, err := NewWorkingSet(WorkingSetConfig{
		JobID:      "job-async",
		Backend:    b,
		ParkPolicy: schema.ParkSuspendAsync,
	})
	if err != nil {
		t.Fatal(err)
	}

	ws.BeginStep()
	e1, err := ws.Acquire(ctx, 1, 0, "inst-a", "img-a")
	if err != nil {
		t.Fatal(err)
	}
	if err := ws.Release(ctx, e1); err != nil {
		t.Fatal(err)
	}
	// Prewarm img-b for step 2 while we pause between steps.
	ws.Prewarm(ctx, 2, []string{"img-b"})
	barrierAt := time.Now()
	time.Sleep(15 * time.Millisecond)
	resumeAt := time.Now()

	overlapMS, err := ws.AwaitAsyncSuspends(ctx, barrierAt, resumeAt)
	if err != nil || overlapMS <= 0 {
		t.Fatalf("AwaitAsyncSuspends = (%v, %v), want >0 overlap", overlapMS, err)
	}
	leadMS, err := ws.WaitPrewarm(ctx, 2, resumeAt)
	if err != nil || leadMS <= 0 {
		t.Fatalf("WaitPrewarm = (%v, %v), want >0 lead margin", leadMS, err)
	}

	// Step 2 should get a hit on both img-a and prewarmed img-b!
	ws.BeginStep()
	hA, err := ws.Acquire(ctx, 2, 0, "inst-a", "img-a")
	if err != nil || !hA.Hit {
		t.Fatalf("step 2 img-a hit=%v err=%v", hA.Hit, err)
	}
	hB, err := ws.Acquire(ctx, 2, 1, "inst-b", "img-b")
	if err != nil || !hB.Hit {
		t.Fatalf("step 2 prewarmed img-b hit=%v err=%v", hB.Hit, err)
	}
	_ = ws.Release(ctx, hA)
	_ = ws.Release(ctx, hB)
	_, _ = ws.AwaitAsyncSuspends(ctx, time.Now(), time.Now())
	_ = ws.Close(ctx)
}

func TestRunScheduledStepsRecordsLabelsAndTrajectoryMix(t *testing.T) {
	ctx := context.Background()
	fb := &wsFakeBackend{}
	var buf bytes.Buffer
	rec := NewRecorder(&buf)

	tA := &Trajectory{Name: "traj-a", Steps: []Step{{Name: "a", Run: "true"}}}
	tB := &Trajectory{Name: "traj-b", Steps: []Step{{Name: "b", Run: "true"}}}
	insts := []Instance{
		{InstanceID: "inst-1", Image: "reg/img-1@" + goodDigest},
		{InstanceID: "inst-2", Image: "reg/img-2@" + goodDigest},
	}

	sum, err := Run(ctx, Config{
		RunID:        "ws-mix",
		Backend:      fb,
		Instances:    insts,
		Trajectories: []*Trajectory{tA, tB},
		Mode:         ModeSteps,
		Park:         ParkSuspendSync,
		Steps:        2,
		TasksPerStep: 2,
		GroupSize:    2,
		Width:        2,
		Seed:         42,
		Labels:       map[string]string{"arm": "candidate"},
		Recorder:     rec,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if sum.Tasks != 8 || sum.Failed != 0 {
		t.Fatalf("Summary = %+v, want 8 tasks and 0 failed", sum)
	}

	evs, err := ReadEvents(&buf)
	if err != nil {
		t.Fatalf("ReadEvents: %v", err)
	}
	seenTraj := map[string]int{}
	var runLabels map[string]string
	for _, ev := range evs {
		switch ev.Event {
		case EventTask:
			seenTraj[ev.Name]++
		case EventRun:
			runLabels = ev.Labels
		}
	}
	if seenTraj["traj-a"] == 0 || seenTraj["traj-b"] == 0 {
		t.Fatalf("expected both trajectories in mix to be drawn, got %v", seenTraj)
	}
	if runLabels["arm"] != "candidate" || runLabels["trajectory"] != "traj-a,traj-b" || runLabels["mode"] != "steps" {
		t.Fatalf("EventRun Labels = %v, want arm=candidate, trajectory=traj-a,traj-b, mode=steps", runLabels)
	}
}

func BenchmarkWorkingSetThroughput(b *testing.B) {
	ctx := context.Background()
	fb := &wsFakeBackend{}
	ws, err := NewWorkingSet(WorkingSetConfig{
		JobID:      "bench",
		Backend:    fb,
		ParkPolicy: schema.ParkSuspendSync,
		MaxParked:  64,
	})
	if err != nil {
		b.Fatal(err)
	}
	images := make([]string, 32)
	for i := range images {
		images[i] = fmt.Sprintf("reg/img-%02d@%s", i, goodDigest)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		img := images[i%len(images)]
		e, err := ws.Acquire(ctx, i/len(images)+1, i%len(images), "inst", img)
		if err != nil {
			b.Fatal(err)
		}
		if err := ws.Release(ctx, e); err != nil {
			b.Fatal(err)
		}
	}
}
