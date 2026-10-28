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
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"slices"

	"github.com/agent-substrate/env/internal/benchmarking/schema"
)

// ScheduleSource identifies how a job's per-step task list is produced.
type ScheduleSource string

const (
	// ScheduleSourceExplicit uses a caller-supplied resolved ScheduledJob directly.
	ScheduleSourceExplicit ScheduleSource = "explicit"
	// ScheduleSourceSampled deterministically samples tasks_per_step instances
	// per step using a seeded PCG PRNG and expands each by group_size.
	ScheduleSourceSampled ScheduleSource = "sampled"
	// ScheduleSourceFixed repeats the first tasks_per_step instances of the
	// dataset on every step (100% working-set overlap across steps).
	ScheduleSourceFixed ScheduleSource = "fixed"
)

// JobScheduleSpec describes how to resolve one job's schedule.
type JobScheduleSpec struct {
	JobID         string
	Source        ScheduleSource
	Seed          uint64
	StartOffsetMS float64
	StepPauseMS   float64
	ParkPolicy    schema.ParkPolicy
	Trajectory    string
	Steps         int
	TasksPerStep  int
	GroupSize     int
	Instances     []Instance
	ExplicitJob   *schema.ScheduledJob
}

// ResolveScheduleOptions configures multi-job deterministic schedule resolution.
type ResolveScheduleOptions struct {
	ScenarioName string
	Seed         int64
	Jobs         []JobScheduleSpec
}

// ResolveSchedule resolves a deterministic v1 schema.Schedule from the given options.
func ResolveSchedule(opts ResolveScheduleOptions) (*schema.Schedule, error) {
	if opts.ScenarioName == "" {
		return nil, errors.New("swebench: scenario_name is required")
	}
	if len(opts.Jobs) == 0 {
		return nil, errors.New("swebench: at least one job is required")
	}
	seed := opts.Seed
	if seed == 0 && len(opts.Jobs) > 0 {
		seed = int64(opts.Jobs[0].Seed)
	}

	out := &schema.Schedule{
		SchemaVersion: schema.SchemaVersion,
		ScenarioName:  opts.ScenarioName,
		Seed:          seed,
		Jobs:          make([]schema.ScheduledJob, 0, len(opts.Jobs)),
	}

	for i, js := range opts.Jobs {
		resolved, err := resolveJobSchedule(i, js)
		if err != nil {
			return nil, err
		}
		out.Jobs = append(out.Jobs, resolved)
	}

	if err := out.Validate(); err != nil {
		return nil, err
	}
	return out, nil
}

// ResolveSingleJobSchedule resolves a single-job schedule for swebench-loader.
func ResolveSingleJobSchedule(jobID string, seed uint64, park schema.ParkPolicy, steps, tasksPerStep, groupSize int, instances []Instance) (*schema.Schedule, error) {
	if tasksPerStep <= 0 {
		tasksPerStep = len(instances)
	}
	if groupSize <= 0 {
		groupSize = 1
	}
	return ResolveSchedule(ResolveScheduleOptions{
		ScenarioName: jobID,
		Seed:         int64(seed),
		Jobs: []JobScheduleSpec{
			{
				JobID:        jobID,
				Source:       ScheduleSourceSampled,
				Seed:         seed,
				ParkPolicy:   park,
				Steps:        steps,
				TasksPerStep: tasksPerStep,
				GroupSize:    groupSize,
				Instances:    instances,
			},
		},
	})
}

func resolveJobSchedule(idx int, spec JobScheduleSpec) (schema.ScheduledJob, error) {
	src := spec.Source
	if src == "" {
		src = ScheduleSourceSampled
	}

	if src == ScheduleSourceExplicit {
		if spec.ExplicitJob == nil {
			return schema.ScheduledJob{}, fmt.Errorf("swebench: jobs[%d] (%s): explicit source requires ExplicitJob", idx, spec.JobID)
		}
		cp := *spec.ExplicitJob
		if spec.JobID != "" {
			cp.Job = spec.JobID
		}
		if spec.ParkPolicy != "" {
			norm, err := schema.NormalizeParkPolicy(spec.ParkPolicy)
			if err != nil {
				return schema.ScheduledJob{}, fmt.Errorf("swebench: job %s: %w", cp.Job, err)
			}
			cp.ParkPolicy = norm
		}
		populatePrefetchImages(cp.Steps)
		return cp, nil
	}

	if spec.JobID == "" {
		return schema.ScheduledJob{}, fmt.Errorf("swebench: jobs[%d]: job name is required", idx)
	}
	var normPark schema.ParkPolicy
	if spec.ParkPolicy != "" {
		var err error
		normPark, err = schema.NormalizeParkPolicy(spec.ParkPolicy)
		if err != nil {
			return schema.ScheduledJob{}, fmt.Errorf("swebench: job %s: %w", spec.JobID, err)
		}
	}
	if spec.Steps < 1 {
		return schema.ScheduledJob{}, fmt.Errorf("swebench: job %s: steps must be >= 1, got %d", spec.JobID, spec.Steps)
	}
	if spec.TasksPerStep < 1 {
		return schema.ScheduledJob{}, fmt.Errorf("swebench: job %s: tasks_per_step must be >= 1, got %d", spec.JobID, spec.TasksPerStep)
	}
	if spec.GroupSize < 1 {
		return schema.ScheduledJob{}, fmt.Errorf("swebench: job %s: group_size must be >= 1, got %d", spec.JobID, spec.GroupSize)
	}
	if len(spec.Instances) == 0 {
		return schema.ScheduledJob{}, fmt.Errorf("swebench: job %s: dataset instances cannot be empty", spec.JobID)
	}

	steps := make([]schema.ScheduledStep, spec.Steps)
	switch src {
	case ScheduleSourceSampled:
		rng := rand.New(rand.NewPCG(spec.Seed, spec.Seed^0x9e3779b97f4a7c15))
		for s := 0; s < spec.Steps; s++ {
			picked := sampleInstancesForStep(rng, spec.Instances, spec.TasksPerStep)
			steps[s] = buildScheduledStep(s+1, picked, spec.GroupSize, spec.Trajectory, spec.StepPauseMS)
		}
	case ScheduleSourceFixed:
		picked := fixedInstancesForStep(spec.Instances, spec.TasksPerStep)
		for s := 0; s < spec.Steps; s++ {
			steps[s] = buildScheduledStep(s+1, picked, spec.GroupSize, spec.Trajectory, spec.StepPauseMS)
		}
	default:
		return schema.ScheduledJob{}, fmt.Errorf("swebench: job %s: unsupported schedule source %q", spec.JobID, src)
	}

	populatePrefetchImages(steps)
	return schema.ScheduledJob{
		Job:           spec.JobID,
		StartOffsetMS: spec.StartOffsetMS,
		ParkPolicy:    normPark,
		Steps:         steps,
	}, nil
}

func sampleInstancesForStep(rng *rand.Rand, pool []Instance, count int) []Instance {
	n := len(pool)
	out := make([]Instance, 0, count)
	for len(out) < count {
		perm := rng.Perm(n)
		need := count - len(out)
		if need > n {
			need = n
		}
		for i := 0; i < need; i++ {
			out = append(out, pool[perm[i]])
		}
	}
	return out
}

func fixedInstancesForStep(pool []Instance, count int) []Instance {
	out := make([]Instance, count)
	for i := 0; i < count; i++ {
		out[i] = pool[i%len(pool)]
	}
	return out
}

func buildScheduledStep(stepNum int, picked []Instance, groupSize int, trajectory string, stepPauseMS float64) schema.ScheduledStep {
	tasks := make([]schema.ScheduledTask, 0, len(picked)*groupSize)
	for _, inst := range picked {
		for g := 0; g < groupSize; g++ {
			tasks = append(tasks, schema.ScheduledTask{
				InstanceID: inst.InstanceID,
				Image:      inst.Image,
				Trajectory: trajectory,
				Rollout:    g,
			})
		}
	}
	return schema.ScheduledStep{
		Step:        stepNum,
		StepPauseMS: stepPauseMS,
		Tasks:       tasks,
	}
}

// populatePrefetchImages computes Step[s].PrefetchImages = need(s+1) \ active(s)
// in deterministic order of first appearance in Step[s+1].
func populatePrefetchImages(steps []schema.ScheduledStep) {
	for s := 0; s+1 < len(steps); s++ {
		active := make(map[string]bool, len(steps[s].Tasks))
		for _, t := range steps[s].Tasks {
			active[t.Image] = true
		}
		seen := make(map[string]bool, len(steps[s+1].Tasks))
		var prefetch []string
		for _, t := range steps[s+1].Tasks {
			if !active[t.Image] && !seen[t.Image] {
				seen[t.Image] = true
				prefetch = append(prefetch, t.Image)
			}
		}
		steps[s].PrefetchImages = prefetch
	}
	if len(steps) > 0 {
		steps[len(steps)-1].PrefetchImages = nil
	}
}

// WriteSchedule validates and writes a Schedule to path as formatted JSON.
func WriteSchedule(path string, sched *schema.Schedule) error {
	if sched == nil {
		return errors.New("swebench: nil schedule")
	}
	if err := sched.Validate(); err != nil {
		return err
	}
	data, err := json.MarshalIndent(sched, "", "  ")
	if err != nil {
		return fmt.Errorf("swebench: marshal schedule: %w", err)
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

// ReadSchedule reads and strictly validates a Schedule from path.
func ReadSchedule(path string) (*schema.Schedule, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("swebench: read schedule: %w", err)
	}
	sched, err := schema.DecodeJSONStrict[schema.Schedule](bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("swebench: decode schedule %s: %w", path, err)
	}
	return sched, nil
}

// InstancesByID indexes a slice of Instances by InstanceID.
func InstancesByID(instances []Instance) map[string]Instance {
	m := make(map[string]Instance, len(instances))
	for _, in := range instances {
		m[in.InstanceID] = in
	}
	return m
}

// UniqueScheduleImages returns all unique images referenced by sched in sorted order.
func UniqueScheduleImages(sched *schema.Schedule) []string {
	seen := map[string]bool{}
	var out []string
	for _, job := range sched.Jobs {
		for _, st := range job.Steps {
			for _, t := range st.Tasks {
				if !seen[t.Image] {
					seen[t.Image] = true
					out = append(out, t.Image)
				}
			}
		}
	}
	slices.Sort(out)
	return out
}
