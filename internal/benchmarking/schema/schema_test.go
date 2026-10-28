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
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestGoldenScheduleRoundTripAndSchemaValidation(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "schedule.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateAgainstJSONSchema("schedule.schema.json", raw); err != nil {
		t.Fatalf("JSON schema validation failed: %v", err)
	}
	sched, err := DecodeJSONStrict[Schedule](bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("DecodeJSONStrict: %v", err)
	}
	out, err := json.Marshal(sched)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if err := ValidateAgainstJSONSchema("schedule.schema.json", out); err != nil {
		t.Fatalf("re-encoded JSON schema validation failed: %v", err)
	}
	roundTrip, err := DecodeJSONStrict[Schedule](bytes.NewReader(out))
	if err != nil {
		t.Fatalf("round-trip DecodeJSONStrict: %v", err)
	}
	if !reflect.DeepEqual(roundTrip, sched) {
		t.Fatalf("round-trip mismatch: got %+v, want %+v", roundTrip, sched)
	}
}

func TestGoldenPlacementRoundTripAndSchemaValidation(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "placement.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := ValidateAgainstJSONSchema("placement.schema.json", raw); err != nil {
		t.Fatalf("JSON schema validation failed: %v", err)
	}
	plan, err := DecodeJSONStrict[PlacementPlan](bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("DecodeJSONStrict: %v", err)
	}
	out, err := json.Marshal(plan)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if err := ValidateAgainstJSONSchema("placement.schema.json", out); err != nil {
		t.Fatalf("re-encoded JSON schema validation failed: %v", err)
	}
	roundTrip, err := DecodeJSONStrict[PlacementPlan](bytes.NewReader(out))
	if err != nil {
		t.Fatalf("round-trip DecodeJSONStrict: %v", err)
	}
	if !reflect.DeepEqual(roundTrip, plan) {
		t.Fatalf("round-trip mismatch: got %+v, want %+v", roundTrip, plan)
	}
}

func TestStrictDecodingRejectsUnknownFields(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "schedule.json"))
	if err != nil {
		t.Fatal(err)
	}
	mutated := strings.Replace(string(raw), `"scenario_name":`, `"unknown_field": 123, "scenario_name":`, 1)
	if _, err := DecodeJSONStrict[Schedule](strings.NewReader(mutated)); err == nil {
		t.Fatal("DecodeJSONStrict succeeded on schedule.json with unknown_field, want error")
	}
	if err := ValidateAgainstJSONSchema("schedule.schema.json", []byte(mutated)); err == nil {
		t.Fatal("ValidateAgainstJSONSchema succeeded on schedule.json with unknown_field, want error")
	}
}

func TestParkPolicyNormalization(t *testing.T) {
	for _, tc := range []struct {
		in      ParkPolicy
		want    ParkPolicy
		wantErr bool
	}{
		{in: ParkSuspendAsync, want: ParkSuspendAsync},
		{in: ParkSuspendSync, want: ParkSuspendSync},
		{in: ParkLegacySuspend, want: ParkSuspendSync},
		{in: ParkFork, want: ParkFork},
		{in: ParkDelete, want: ParkDelete},
		{in: ParkNone, want: ParkNone},
		{in: "bogus", wantErr: true},
	} {
		got, err := NormalizeParkPolicy(tc.in)
		if tc.wantErr {
			if err == nil {
				t.Errorf("NormalizeParkPolicy(%q) succeeded, want error", tc.in)
			}
			continue
		}
		if err != nil || got != tc.want {
			t.Errorf("NormalizeParkPolicy(%q) = (%q, %v), want (%q, nil)", tc.in, got, err, tc.want)
		}
	}
}

func TestScheduleValidationRejectsInvalidFields(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "schedule.json"))
	if err != nil {
		t.Fatal(err)
	}
	base, err := DecodeJSONStrict[Schedule](bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}

	badVer := *base
	badVer.SchemaVersion = "v0"
	if err := badVer.Validate(); err == nil {
		t.Error("expected error for wrong schema_version")
	}

	badStep := *base
	badStep.Jobs = append([]ScheduledJob(nil), base.Jobs...)
	badStep.Jobs[0].Steps = append([]ScheduledStep(nil), base.Jobs[0].Steps...)
	badStep.Jobs[0].Steps[0].Step = 0
	if err := badStep.Validate(); err == nil {
		t.Error("expected error for invalid step number 0")
	}
}
