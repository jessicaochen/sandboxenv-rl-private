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

package schema

import (
	"errors"
	"fmt"
)

// Schedule is the deterministic resolved per-step task schedule (schedule.json).
type Schedule struct {
	SchemaVersion string         `json:"schema_version"`
	ScenarioName  string         `json:"scenario_name"`
	Seed          int64          `json:"seed,omitempty"`
	Jobs          []ScheduledJob `json:"jobs"`
}

// Validate checks Schedule invariants.
func (s Schedule) Validate() error {
	if s.SchemaVersion != SchemaVersion {
		return fmt.Errorf("schedule: unsupported schema_version %q (want %q)", s.SchemaVersion, SchemaVersion)
	}
	if s.ScenarioName == "" {
		return errors.New("schedule: scenario_name is required")
	}
	if len(s.Jobs) == 0 {
		return errors.New("schedule: at least one scheduled job is required")
	}
	seen := make(map[string]bool, len(s.Jobs))
	for i, j := range s.Jobs {
		if err := j.Validate(); err != nil {
			return fmt.Errorf("schedule job[%d]: %w", i, err)
		}
		if seen[j.Job] {
			return fmt.Errorf("schedule: duplicate job %q", j.Job)
		}
		seen[j.Job] = true
	}
	return nil
}

// ScheduledJob holds the resolved step list for one job.
type ScheduledJob struct {
	Job           string          `json:"job"`
	StartOffsetMS float64         `json:"start_offset_ms,omitempty"`
	ParkPolicy    ParkPolicy      `json:"park_policy,omitempty"`
	Steps         []ScheduledStep `json:"steps"`
}

// Validate checks ScheduledJob invariants.
func (j ScheduledJob) Validate() error {
	if j.Job == "" {
		return errors.New("job name is required")
	}
	if j.StartOffsetMS < 0 {
		return fmt.Errorf("job %q: start_offset_ms must be >= 0, got %f", j.Job, j.StartOffsetMS)
	}
	if j.ParkPolicy != "" {
		if _, err := NormalizeParkPolicy(j.ParkPolicy); err != nil {
			return fmt.Errorf("job %q: %w", j.Job, err)
		}
	}
	if len(j.Steps) == 0 {
		return fmt.Errorf("job %q: at least one step is required", j.Job)
	}
	for i, st := range j.Steps {
		if err := st.Validate(); err != nil {
			return fmt.Errorf("job %q step[%d]: %w", j.Job, i, err)
		}
	}
	return nil
}

// ScheduledStep holds the tasks and prefetch list for a single step.
type ScheduledStep struct {
	Step           int             `json:"step"`
	StepPauseMS    float64         `json:"step_pause_ms,omitempty"`
	Tasks          []ScheduledTask `json:"tasks"`
	PrefetchImages []string        `json:"prefetch_images,omitempty"`
}

// Validate checks ScheduledStep invariants.
func (s ScheduledStep) Validate() error {
	if s.Step <= 0 {
		return fmt.Errorf("step must be >= 1, got %d", s.Step)
	}
	if s.StepPauseMS < 0 {
		return fmt.Errorf("step %d: step_pause_ms must be >= 0, got %f", s.Step, s.StepPauseMS)
	}
	if len(s.Tasks) == 0 {
		return fmt.Errorf("step %d: at least one task is required", s.Step)
	}
	for i, t := range s.Tasks {
		if err := t.Validate(); err != nil {
			return fmt.Errorf("step %d task[%d]: %w", s.Step, i, err)
		}
	}
	return nil
}

// ScheduledTask specifies one task rollout within a ScheduledStep.
type ScheduledTask struct {
	InstanceID string `json:"instance_id"`
	Image      string `json:"image"`
	Trajectory string `json:"trajectory,omitempty"`
	Rollout    int    `json:"rollout,omitempty"`
}

// Validate checks ScheduledTask invariants.
func (t ScheduledTask) Validate() error {
	if t.InstanceID == "" {
		return errors.New("instance_id is required")
	}
	if t.Image == "" {
		return errors.New("image is required")
	}
	if t.Rollout < 0 {
		return fmt.Errorf("rollout must be >= 0, got %d", t.Rollout)
	}
	return nil
}

// PlacementPlan is an optional separate artifact (placement.json) for scripting
// explicit sandbox-assignment decisions keyed by (step, instance_id, rollout).
type PlacementPlan struct {
	SchemaVersion string             `json:"schema_version"`
	ScenarioName  string             `json:"scenario_name"`
	Jobs          []JobPlacementPlan `json:"jobs"`
}

// Validate checks PlacementPlan invariants.
func (p PlacementPlan) Validate() error {
	if p.SchemaVersion != SchemaVersion {
		return fmt.Errorf("placement: unsupported schema_version %q (want %q)", p.SchemaVersion, SchemaVersion)
	}
	if p.ScenarioName == "" {
		return errors.New("placement: scenario_name is required")
	}
	if len(p.Jobs) == 0 {
		return errors.New("placement: at least one job placement plan is required")
	}
	for i, j := range p.Jobs {
		if err := j.Validate(); err != nil {
			return fmt.Errorf("placement job[%d]: %w", i, err)
		}
	}
	return nil
}

// JobPlacementPlan holds scripted sandbox assignments for one job.
type JobPlacementPlan struct {
	Job         string          `json:"job"`
	Assignments []TaskPlacement `json:"assignments"`
}

// Validate checks JobPlacementPlan invariants.
func (j JobPlacementPlan) Validate() error {
	if j.Job == "" {
		return errors.New("job name is required")
	}
	if len(j.Assignments) == 0 {
		return fmt.Errorf("job %q: at least one assignment is required", j.Job)
	}
	for i, a := range j.Assignments {
		if err := a.Validate(); err != nil {
			return fmt.Errorf("job %q assignment[%d]: %w", j.Job, i, err)
		}
	}
	return nil
}

// TaskPlacement maps a (step, instance_id, rollout) tuple to a scripted environment action.
type TaskPlacement struct {
	Step         int    `json:"step"`
	InstanceID   string `json:"instance_id"`
	Rollout      int    `json:"rollout,omitempty"`
	EnvID        string `json:"env_id"`
	ReuseEnvID   string `json:"reuse_env_id,omitempty"`
	FromSnapshot string `json:"from_snapshot,omitempty"`
	RetireAfter  bool   `json:"retire_after,omitempty"`
}

// Validate checks TaskPlacement invariants.
func (p TaskPlacement) Validate() error {
	if p.Step <= 0 {
		return fmt.Errorf("step must be >= 1, got %d", p.Step)
	}
	if p.InstanceID == "" {
		return errors.New("instance_id is required")
	}
	if p.Rollout < 0 {
		return fmt.Errorf("rollout must be >= 0, got %d", p.Rollout)
	}
	if p.EnvID == "" {
		return errors.New("env_id is required")
	}
	return nil
}
