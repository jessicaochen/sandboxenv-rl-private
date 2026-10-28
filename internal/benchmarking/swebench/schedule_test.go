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
	"fmt"
	"path/filepath"
	"testing"

	"github.com/agent-substrate/env/internal/benchmarking/schema"
)

func makeScheduleDataset(n int) []Instance {
	out := make([]Instance, n)
	for i := 0; i < n; i++ {
		out[i] = Instance{
			InstanceID: fmt.Sprintf("repo__task-%03d", i),
			Image:      fmt.Sprintf("us-central1-docker.pkg.dev/proj/swe/task-%03d@%s", i, goodDigest),
			EvalScript: "#!/bin/sh\nexit 0",
			LogParser:  "parse_log_pytest",
		}
	}
	return out
}

func TestResolveScheduleDeterminismAndPrefetchSetDiff(t *testing.T) {
	dataset := makeScheduleDataset(8)

	opts := ResolveScheduleOptions{
		ScenarioName: "sched-det-01",
		Seed:         42,
		Jobs: []JobScheduleSpec{
			{
				JobID:        "job-sampled",
				Source:       ScheduleSourceSampled,
				Seed:         42,
				ParkPolicy:   schema.ParkSuspendAsync,
				Steps:        4,
				TasksPerStep: 3,
				GroupSize:    2,
				Instances:    dataset,
			},
			{
				JobID:        "job-fixed",
				Source:       ScheduleSourceFixed,
				Seed:         99,
				ParkPolicy:   schema.ParkSuspendSync,
				Steps:        4,
				TasksPerStep: 2,
				GroupSize:    2,
				Instances:    dataset,
			},
		},
	}

	s1, err := ResolveSchedule(opts)
	if err != nil {
		t.Fatalf("ResolveSchedule(1): %v", err)
	}
	s2, err := ResolveSchedule(opts)
	if err != nil {
		t.Fatalf("ResolveSchedule(2): %v", err)
	}

	b1, err := json.Marshal(s1)
	if err != nil {
		t.Fatal(err)
	}
	b2, err := json.Marshal(s2)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(b1, b2) {
		t.Fatal("same seed produced non-identical schedule JSON")
	}
	if err := schema.ValidateAgainstJSONSchema("schedule.schema.json", b1); err != nil {
		t.Fatalf("resolved schedule failed JSON Schema validation: %v", err)
	}

	// Changing seed produces a different sampled schedule.
	optsDiff := opts
	optsDiff.Jobs = append([]JobScheduleSpec(nil), opts.Jobs...)
	optsDiff.Jobs[0].Seed = 43
	s3, err := ResolveSchedule(optsDiff)
	if err != nil {
		t.Fatalf("ResolveSchedule(diff seed): %v", err)
	}
	b3, _ := json.Marshal(s3)
	if bytes.Equal(b1, b3) {
		t.Fatal("different seed produced identical schedule JSON")
	}

	// Verify group_size expansion and prefetch_images = need(s+1) \ active(s).
	sampledJob := s1.Jobs[0]
	for sIdx, st := range sampledJob.Steps {
		if len(st.Tasks) != 6 {
			t.Fatalf("step %d: got %d tasks, want 6 (3*2)", st.Step, len(st.Tasks))
		}
		for i := 0; i < len(st.Tasks); i += 2 {
			if st.Tasks[i].InstanceID != st.Tasks[i+1].InstanceID || st.Tasks[i].Rollout != 0 || st.Tasks[i+1].Rollout != 1 {
				t.Fatalf("step %d pair %d invalid group expansion: %+v vs %+v", st.Step, i, st.Tasks[i], st.Tasks[i+1])
			}
		}
		if sIdx+1 < len(sampledJob.Steps) {
			active := map[string]bool{}
			for _, tk := range st.Tasks {
				active[tk.Image] = true
			}
			for _, pf := range st.PrefetchImages {
				if active[pf] {
					t.Fatalf("step %d prefetch image %s is already active in step %d", st.Step, pf, st.Step)
				}
			}
		} else if len(st.PrefetchImages) != 0 {
			t.Fatalf("last step must have empty prefetch_images, got %v", st.PrefetchImages)
		}
	}

	// Fixed job has 100% overlap across steps, so prefetch_images is always empty.
	fixedJob := s1.Jobs[1]
	for _, st := range fixedJob.Steps {
		if len(st.PrefetchImages) != 0 {
			t.Fatalf("fixed job step %d expected 0 prefetch_images, got %v", st.Step, st.PrefetchImages)
		}
	}

	// Round-trip WriteSchedule and ReadSchedule.
	path := filepath.Join(t.TempDir(), "schedule.json")
	if err := WriteSchedule(path, s1); err != nil {
		t.Fatalf("WriteSchedule: %v", err)
	}
	readBack, err := ReadSchedule(path)
	if err != nil {
		t.Fatalf("ReadSchedule: %v", err)
	}
	if len(readBack.Jobs) != len(s1.Jobs) || readBack.ScenarioName != s1.ScenarioName {
		t.Fatalf("ReadSchedule mismatch: got %+v", readBack)
	}
}
