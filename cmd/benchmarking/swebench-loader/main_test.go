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

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/agent-substrate/env/internal/benchmarking/swebench"
)

func TestResolveAuthToken(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "token")
	if err := os.WriteFile(file, []byte("  from-file\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	empty := filepath.Join(dir, "empty")
	if err := os.WriteFile(empty, []byte(" \n"), 0o600); err != nil {
		t.Fatal(err)
	}
	env := func(v string) func(string) string {
		return func(k string) string {
			if k == authTokenEnv {
				return v
			}
			return ""
		}
	}
	for _, tc := range []struct {
		name, file, env, want, err string
	}{
		{name: "none", want: ""},
		{name: "env", env: " from-env ", want: "from-env"},
		{name: "file wins over env", file: file, env: "from-env", want: "from-file"},
		{name: "empty file", file: empty, err: "is empty"},
		{name: "missing file", file: filepath.Join(dir, "nope"), err: "reading -auth-token-file"},
	} {
		got, err := resolveAuthToken(tc.file, env(tc.env))
		if tc.err != "" {
			if err == nil || !strings.Contains(err.Error(), tc.err) {
				t.Errorf("%s: err = %v, want %q", tc.name, err, tc.err)
			}
			continue
		}
		if err != nil || got != tc.want {
			t.Errorf("%s: got %q, %v; want %q", tc.name, got, err, tc.want)
		}
	}
}

func TestLeadingFiles(t *testing.T) {
	for _, tc := range []struct {
		args, files, rest []string
	}{
		{nil, nil, nil},
		{[]string{"a.jsonl"}, []string{"a.jsonl"}, nil},
		{[]string{"a.jsonl", "b.jsonl", "-json", "o.json"}, []string{"a.jsonl", "b.jsonl"}, []string{"-json", "o.json"}},
		{[]string{"-json", "o.json", "a.jsonl"}, nil, []string{"-json", "o.json", "a.jsonl"}},
	} {
		files, rest := leadingFiles(tc.args)
		if len(files) == 0 {
			files = nil
		}
		if len(rest) == 0 {
			rest = nil
		}
		if !reflect.DeepEqual(files, tc.files) || !reflect.DeepEqual(rest, tc.rest) {
			t.Errorf("leadingFiles(%q) = %q, %q; want %q, %q", tc.args, files, rest, tc.files, tc.rest)
		}
	}
}

func TestReportCmdRejectsFilesAfterFlags(t *testing.T) {
	err := reportCmd([]string{"a.jsonl", "-json", "o.json", "b.jsonl"}, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "before the flags") || !strings.Contains(err.Error(), "b.jsonl") {
		t.Fatalf("err = %v, want the files-before-flags error naming b.jsonl", err)
	}
}

type e2eFakeBackend struct{}

func (e2eFakeBackend) Create(_ context.Context, id, _ string) (swebench.Sandbox, error) {
	return e2eFakeSandbox{id: id}, nil
}

func (e2eFakeBackend) Delete(_ context.Context, _ string) error { return nil }

type e2eFakeSandbox struct {
	id string
}

func (s e2eFakeSandbox) Run(_ context.Context, command string, _ time.Duration) (swebench.Result, error) {
	if strings.Contains(command, "/tmp/grade.py") {
		return swebench.Result{
			ExitCode: 0,
			Stdout:   `{"valid":true,"resolved":true,"f2p":"1/1","p2p":"2/2"}` + "\n",
		}, nil
	}
	if strings.HasPrefix(command, "echo ") {
		return swebench.Result{
			ExitCode: 0,
			Stdout:   strings.TrimPrefix(command, "echo ") + "\n",
		}, nil
	}
	return swebench.Result{ExitCode: 0, Stdout: "ok\n"}, nil
}

func (e2eFakeSandbox) WriteFile(_ context.Context, _ string, _ []byte, _ fs.FileMode) error {
	return nil
}

func (e2eFakeSandbox) Suspend(_ context.Context) error { return nil }
func (e2eFakeSandbox) Delete(_ context.Context) error  { return nil }

func TestEndToEndSWEBenchAndCommandDatasets(t *testing.T) {
	const digest = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
	dir := t.TempDir()

	testBackendOverride = e2eFakeBackend{}
	t.Cleanup(func() { testBackendOverride = nil })

	// 1. End-to-end SWE-bench manifest run + report.
	sweManifest := filepath.Join(dir, "swebench.jsonl")
	sweRows := strings.Join([]string{
		`{"instance_id":"astropy__astropy-1","repo":"astropy/astropy","image":"reg/swe@` + digest + `","patch":"diff","eval_script":"#!/bin/sh\nexit 0","log_parser":"parse_log_pytest"}`,
		`{"instance_id":"django__django-2","repo":"django/django","image":"reg/swe@` + digest + `","patch":"diff","eval_script":"#!/bin/sh\nexit 0","log_parser":"parse_log_django"}`,
	}, "\n") + "\n"
	if err := os.WriteFile(sweManifest, []byte(sweRows), 0o644); err != nil {
		t.Fatal(err)
	}
	sweEvents := filepath.Join(dir, "swe-events.jsonl")
	sweSummary := filepath.Join(dir, "swe-summary.json")
	if err := runCmd([]string{
		"-dataset", sweManifest,
		"-mode", "eval",
		"-width", "2",
		"-run-id", "swe-e2e",
		"-out", sweEvents,
	}); err != nil {
		t.Fatalf("runCmd(swebench): %v", err)
	}
	if err := reportCmd([]string{sweEvents, "-json", sweSummary}); err != nil {
		t.Fatalf("reportCmd(swebench): %v", err)
	}
	repSWE, err := summarizeFile(sweEvents)
	if err != nil {
		t.Fatalf("summarizeFile(sweEvents): %v", err)
	}
	if repSWE.Tasks != 2 || repSWE.Failed != 0 || repSWE.Rewarded != 2 || repSWE.RewardRate != 1.0 {
		t.Fatalf("unexpected SWE-bench report: %+v", repSWE)
	}

	// 2. End-to-end Command manifest run + report.
	cmdManifest := filepath.Join(dir, "command.jsonl")
	cmdRows := strings.Join([]string{
		`{"kind":"command","instance_id":"cmd-alpha","image":"reg/cmd@` + digest + `","command":"echo alpha-ok","setup_command":"mkdir -p /tmp/work","reset_command":"rm -rf /tmp/work/*","expected_stdout":"alpha-ok","expected_exit_code":0}`,
		`{"kind":"command","instance_id":"cmd-beta","image":"reg/cmd@` + digest + `","command":"echo beta-ok","expected_stdout":"beta-ok","expected_exit_code":0}`,
	}, "\n") + "\n"
	if err := os.WriteFile(cmdManifest, []byte(cmdRows), 0o644); err != nil {
		t.Fatal(err)
	}
	cmdEvents := filepath.Join(dir, "cmd-events.jsonl")
	cmdSummary := filepath.Join(dir, "cmd-summary.json")
	if err := runCmd([]string{
		"-dataset", cmdManifest,
		"-mode", "steps",
		"-rollouts", "1",
		"-steps", "2",
		"-park", "suspend",
		"-width", "2",
		"-run-id", "cmd-e2e",
		"-out", cmdEvents,
	}); err != nil {
		t.Fatalf("runCmd(command): %v", err)
	}
	if err := reportCmd([]string{cmdEvents, "-json", cmdSummary}); err != nil {
		t.Fatalf("reportCmd(command): %v", err)
	}
	repCmd, err := summarizeFile(cmdEvents)
	if err != nil {
		t.Fatalf("summarizeFile(cmdEvents): %v", err)
	}
	if repCmd.Tasks != 4 || repCmd.Failed != 0 || repCmd.Rewarded != 4 || repCmd.RewardRate != 1.0 {
		t.Fatalf("unexpected Command report: %+v", repCmd)
	}

	// 3. End-to-end resolve + scheduled WorkingSet run + report.
	schedPath := filepath.Join(dir, "schedule.json")
	if err := resolveCmd([]string{
		"-dataset", cmdManifest,
		"-steps", "3",
		"-tasks-per-step", "1",
		"-group-size", "2",
		"-park", "suspend-async",
		"-seed", "42",
		"-run-id", "sched-e2e",
		"-out", schedPath,
	}); err != nil {
		t.Fatalf("resolveCmd: %v", err)
	}
	schedEvents := filepath.Join(dir, "sched-events.jsonl")
	if err := runCmd([]string{
		"-dataset", cmdManifest,
		"-mode", "steps",
		"-schedule", schedPath,
		"-park", "suspend-async",
		"-step-pause", "5ms",
		"-prefetch",
		"-width", "2",
		"-run-id", "sched-e2e",
		"-out", schedEvents,
	}); err != nil {
		t.Fatalf("runCmd(schedule): %v", err)
	}
	repSched, err := summarizeFile(schedEvents)
	if err != nil {
		t.Fatalf("summarizeFile(schedEvents): %v", err)
	}
	if repSched.Tasks != 6 || repSched.Failed != 0 || repSched.Rewarded != 6 {
		t.Fatalf("unexpected scheduled report: %+v", repSched)
	}
	if repSched.WorkingSetHitRate <= 0 || repSched.Created == 0 || repSched.Reused == 0 {
		t.Fatalf("expected non-zero working-set hit rate and reuse in scheduled report: %+v", repSched)
	}
}

func TestLabelFlags(t *testing.T) {
	l := labelFlags{}
	for _, v := range []string{"cluster=east", "note=a=b", "empty="} {
		if err := l.Set(v); err != nil {
			t.Fatalf("Set(%q): %v", v, err)
		}
	}
	if l["cluster"] != "east" || l["note"] != "a=b" || l["empty"] != "" {
		t.Fatalf("labels = %v", l)
	}
	for _, bad := range []string{"cluster=west", "novalue", "=x", "tab=a\tb", "nl=a\nb"} {
		if err := l.Set(bad); err == nil {
			t.Errorf("Set(%q) must fail", bad)
		}
	}
	if got := l.String(); got != "cluster=east,empty=,note=a=b" {
		t.Fatalf("String() = %q", got)
	}
}

func TestReportCmdArmsRejectPositionalFiles(t *testing.T) {
	err := reportCmd([]string{"a.jsonl", "-arm", "pull=p.jsonl"}, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "inside an -arm value") {
		t.Fatalf("err = %v, want the arm-and-files error", err)
	}
	err = reportCmd([]string{"-arm", "pull=p.jsonl", "-top", "3"}, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "not -arm") {
		t.Fatalf("err = %v, want the one-file flags error", err)
	}
	err = reportCmd([]string{"a.jsonl", "-top", "-3"}, io.Discard)
	if err == nil || !strings.Contains(err.Error(), "-top wants 0 or more") {
		t.Fatalf("err = %v, want the negative -top error", err)
	}
}

func TestArmFlags(t *testing.T) {
	var a armFlags
	for _, v := range []string{"stream=s1.jsonl, s2.jsonl", "pull=p.jsonl"} {
		if err := a.Set(v); err != nil {
			t.Fatalf("Set(%q): %v", v, err)
		}
	}
	// Flag order is column order: the arm given first comes first.
	if len(a) != 2 || a[0].name != "stream" || strings.Join(a[0].files, "+") != "s1.jsonl+s2.jsonl" || a[1].name != "pull" {
		t.Fatalf("arms = %+v", a)
	}
	if got := a.String(); got != "stream=s1.jsonl,s2.jsonl pull=p.jsonl" {
		t.Fatalf("String() = %q", got)
	}
	for _, tc := range []struct{ v, err string }{
		{"pull=x.jsonl", "given twice"},
		{"nofile", "want NAME=FILE[,FILE...]"},
		{"empty=", "want NAME=FILE[,FILE...]"},
		{"=x.jsonl", "want NAME=FILE[,FILE...]"},
		{"gap=a.jsonl,,b.jsonl", "empty file name"},
		{"dup=a.jsonl,a.jsonl", "a.jsonl is given twice"},
	} {
		if err := a.Set(tc.v); err == nil || !strings.Contains(err.Error(), tc.err) {
			t.Errorf("Set(%q): err = %v, want %q", tc.v, err, tc.err)
		}
	}
}

// writeEvents writes a complete run: one create and its run event.
func writeEvents(t *testing.T, dir, name, runID string, createMS float64, labels map[string]string) string {
	t.Helper()
	var buf bytes.Buffer
	for _, e := range []swebench.Event{
		{RunID: runID, Event: swebench.EventCreate, DurMS: createMS, OK: true},
		{RunID: runID, Event: swebench.EventRun, Name: "eval", DurMS: 1000, OK: true, Count: 1, Labels: labels},
	} {
		b, err := json.Marshal(e)
		if err != nil {
			t.Fatal(err)
		}
		buf.Write(b)
		buf.WriteByte('\n')
	}
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestReportArms(t *testing.T) {
	dir := t.TempDir()
	a := writeEvents(t, dir, "a.jsonl", "r1", 100, map[string]string{"mode": "eval", "width": "2", "commit": "aaa"})
	b := writeEvents(t, dir, "b.jsonl", "r2", 300, map[string]string{"mode": "eval", "width": "2", "commit": "bbb"})
	c := writeEvents(t, dir, "c.jsonl", "r3", 50, map[string]string{"mode": "eval", "width": "2"})
	wide := writeEvents(t, dir, "wide.jsonl", "r4", 50, map[string]string{"mode": "eval", "width": "3"})
	out := filepath.Join(dir, "arms.json")

	var text bytes.Buffer
	if err := reportCmd([]string{"-arm", "pull=" + a + "," + b, "-arm", "stream=" + c, "-json", out}, &text); err != nil {
		t.Fatal(err)
	}
	flat := strings.Join(strings.Fields(text.String()), " ")
	// Columns in flag order; pull pools two reps: n=2, the median run, the
	// mixed commit shown as such.
	for _, want := range []string{"run pull stream", "reps 2 1", "create 100/300 n=2 50/50 n=1", "commit aaa|bbb -", "width 2 2"} {
		if !strings.Contains(flat, want) {
			t.Errorf("comparison lacks %q:\n%s", want, text.String())
		}
	}
	raw, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var reps []swebench.Report
	if err := json.Unmarshal(raw, &reps); err != nil {
		t.Fatal(err)
	}
	if len(reps) != 2 || reps[0].RunID != "pull" || reps[0].Reps != 2 || strings.Join(reps[0].Runs, ",") != "r1,r2" || reps[1].RunID != "stream" || reps[1].Reps != 1 {
		t.Fatalf("-json: %+v", reps)
	}

	for _, tc := range []struct {
		name string
		args []string
		err  string
	}{
		{"missing file", []string{"-arm", "pull=" + filepath.Join(dir, "nope.jsonl")}, "arm pull"},
		{"empty arm", []string{"-arm", "pull="}, "want NAME=FILE[,FILE...]"},
		{"file twice", []string{"-arm", "pull=" + a + "," + a}, "is given twice"},
		{"other workload", []string{"-arm", "pull=" + a + "," + wide}, "differs from rep 1 in width"},
		{"run in two arms", []string{"-arm", "pull=" + a + "," + b, "-arm", "again=" + a}, "run r1 is in arms pull and again"},
	} {
		if err := reportCmd(tc.args, io.Discard); err == nil || !strings.Contains(err.Error(), tc.err) {
			t.Errorf("%s: err = %v, want %q", tc.name, err, tc.err)
		}
	}
}

func TestOwnLabels(t *testing.T) {
	l := ownLabels("https://host/bucket/swebench.jsonl?X-Goog-Signature=secret", 4, "base", "lists/offline.txt", []string{"x", "y"}, []string{"a", "b"}, map[string]string{"READ_FILES": "10"})
	want := map[string]string{"dataset": "swebench.jsonl", "connections": "4", "base_template": "base",
		"exclude_file": "offline.txt@" + idsDigest([]string{"x", "y"}), "instances": "2@" + idsDigest([]string{"a", "b"}), "set.READ_FILES": "10"}
	if !reflect.DeepEqual(l, want) {
		t.Fatalf("labels = %v, want %v", l, want)
	}
	// The digest tells two lists of one length apart and ignores their order.
	if idsDigest([]string{"a", "b"}) != idsDigest([]string{"b", "a"}) || idsDigest([]string{"a", "b"}) == idsDigest([]string{"a", "c"}) || len(idsDigest(nil)) != 8 {
		t.Errorf("idsDigest: %s %s %s", idsDigest([]string{"a", "b"}), idsDigest([]string{"b", "a"}), idsDigest([]string{"a", "c"}))
	}
	for k := range l {
		if !slices.Contains(swebench.LoaderLabelKeys, k) && !strings.HasPrefix(k, "set.") {
			t.Errorf("label %s is not in LoaderLabelKeys, so report -arm would not compare it", k)
		}
	}
	l = ownLabels("data/swebench.jsonl", 1, "t", "", nil, nil, nil)
	if _, ok := l["exclude_file"]; ok || l["dataset"] != "swebench.jsonl" || len(l) != 3 {
		t.Fatalf("labels without an exclude file or instances = %v", l)
	}
}

func TestAssetsCmd(t *testing.T) {
	var out bytes.Buffer
	if err := assetsCmd([]string{"../../../internal/benchmarking/swebench/trajectories/swebench-gold.yaml"}, &out); err != nil || out.String() != "swebench-grade.py\n" {
		t.Fatalf("trajectory-assets: %q, %v", out.String(), err)
	}
	if err := assetsCmd(nil, io.Discard); err == nil {
		t.Fatal("trajectory-assets without a file must fail")
	}
}
