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

// Package schema defines versioned wire contracts, embedded JSON Schemas, and
// strict decoders for RL benchmark artifacts.
package schema

import (
	"bufio"
	"bytes"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"path"
	"strings"

	"github.com/google/jsonschema-go/jsonschema"
	"sigs.k8s.io/yaml"
)

// SchemaVersion is the wire schema version for all v1 RL benchmark artifacts.
const SchemaVersion = "v1"

// SchemaFS embeds the canonical v1 JSON Schema definitions under v1/*.schema.json.
//
//go:embed v1/*.schema.json
var SchemaFS embed.FS

// Validator is implemented by schema types that enforce semantic invariants.
type Validator interface {
	Validate() error
}

// ParkPolicy enumerates how a step-mode working set parks environments between steps.
type ParkPolicy string

const (
	// ParkSuspendAsync suspends with asynchronous durability (on-node snapshot with background upload).
	ParkSuspendAsync ParkPolicy = "suspend-async"
	// ParkSuspendSync suspends with synchronous durability (waits for durable object storage).
	ParkSuspendSync ParkPolicy = "suspend-sync"
	// ParkFork parks via snapshot/fork semantics.
	ParkFork ParkPolicy = "fork"
	// ParkDelete deletes environments at the step boundary instead of parking.
	ParkDelete ParkPolicy = "delete"
	// ParkNone leaves environments unparked across step boundaries.
	ParkNone ParkPolicy = "none"
	// ParkLegacySuspend is the legacy swebench-loader "suspend" policy, which maps
	// to ParkSuspendSync per DESIGN §5.4 (ate-env-api hardcodes DURABILITY_MODE_SYNCHRONOUS).
	ParkLegacySuspend ParkPolicy = "suspend"
)

// NormalizeParkPolicy validates p against the canonical v1 park policies and
// normalizes legacy "suspend" to ParkSuspendSync per DESIGN §5.4.
func NormalizeParkPolicy(p ParkPolicy) (ParkPolicy, error) {
	switch p {
	case ParkSuspendAsync, ParkSuspendSync, ParkFork, ParkDelete, ParkNone:
		return p, nil
	case ParkLegacySuspend:
		return ParkSuspendSync, nil
	default:
		return "", fmt.Errorf("invalid park policy %q (want %q, %q, %q, %q, %q, or legacy %q)",
			p, ParkSuspendAsync, ParkSuspendSync, ParkFork, ParkDelete, ParkNone, ParkLegacySuspend)
	}
}

func readEmbeddedSchema(name string) (*jsonschema.Schema, error) {
	rel := path.Base(strings.TrimPrefix(name, "v1/"))
	data, err := SchemaFS.ReadFile(path.Join("v1", rel))
	if err != nil {
		return nil, fmt.Errorf("read embedded schema %q: %w", name, err)
	}
	var s jsonschema.Schema
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("unmarshal schema %q: %w", name, err)
	}
	return &s, nil
}

// LoadJSONSchema loads and resolves one of the embedded v1 JSON Schema files
// (for example, "schedule.schema.json" or "v1/schedule.schema.json"), resolving
// any sibling v1/*.schema.json $ref targets from SchemaFS.
func LoadJSONSchema(schemaFile string) (*jsonschema.Resolved, error) {
	s, err := readEmbeddedSchema(schemaFile)
	if err != nil {
		return nil, err
	}
	resolved, err := s.Resolve(&jsonschema.ResolveOptions{
		Loader: func(uri *url.URL) (*jsonschema.Schema, error) {
			return readEmbeddedSchema(path.Base(uri.Path))
		},
	})
	if err != nil {
		return nil, fmt.Errorf("resolve schema %q: %w", schemaFile, err)
	}
	return resolved, nil
}

// ValidateAgainstJSONSchema validates a raw JSON document against an embedded v1 JSON Schema.
func ValidateAgainstJSONSchema(schemaFile string, jsonData []byte) error {
	resolved, err := LoadJSONSchema(schemaFile)
	if err != nil {
		return err
	}
	var doc any
	dec := json.NewDecoder(bytes.NewReader(jsonData))
	if err := dec.Decode(&doc); err != nil {
		return fmt.Errorf("decode json for schema validation: %w", err)
	}
	if dec.More() {
		return errors.New("decode json for schema validation: unexpected trailing data")
	}
	if err := resolved.Validate(doc); err != nil {
		return fmt.Errorf("jsonschema %s: %w", schemaFile, err)
	}
	return nil
}

// ValidateJSONLAgainstJSONSchema validates each non-empty line in r against an embedded v1 JSON Schema.
func ValidateJSONLAgainstJSONSchema(schemaFile string, r io.Reader) error {
	resolved, err := LoadJSONSchema(schemaFile)
	if err != nil {
		return err
	}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<20), 16<<20)
	lineNum := 0
	for sc.Scan() {
		lineNum++
		b := bytes.TrimSpace(sc.Bytes())
		if len(b) == 0 {
			continue
		}
		var doc any
		if err := json.Unmarshal(b, &doc); err != nil {
			return fmt.Errorf("line %d: decode jsonl: %w", lineNum, err)
		}
		if err := resolved.Validate(doc); err != nil {
			return fmt.Errorf("line %d: jsonschema %s: %w", lineNum, schemaFile, err)
		}
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("scan jsonl: %w", err)
	}
	return nil
}

// DecodeJSONStrict decodes a single JSON document from r with DisallowUnknownFields
// and validates the decoded value via its Validate method.
func DecodeJSONStrict[T Validator](r io.Reader) (*T, error) {
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	var out T
	if err := dec.Decode(&out); err != nil {
		return nil, fmt.Errorf("decode json: %w", err)
	}
	if dec.More() {
		return nil, errors.New("decode json: unexpected trailing data")
	}
	if err := out.Validate(); err != nil {
		return nil, err
	}
	return &out, nil
}

// ForEachJSONL streams newline-delimited JSON records from r with DisallowUnknownFields,
// validating each record (if T implements Validator) and invoking fn(lineNum, item)
// without materialising the entire file in memory.
func ForEachJSONL[T any](r io.Reader, fn func(line int, v T) error) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 1<<20), 16<<20)
	lineNum := 0
	for sc.Scan() {
		lineNum++
		b := bytes.TrimSpace(sc.Bytes())
		if len(b) == 0 {
			continue
		}
		dec := json.NewDecoder(bytes.NewReader(b))
		dec.DisallowUnknownFields()
		var item T
		if err := dec.Decode(&item); err != nil {
			return fmt.Errorf("line %d: decode jsonl: %w", lineNum, err)
		}
		if dec.More() {
			return fmt.Errorf("line %d: decode jsonl: unexpected trailing data", lineNum)
		}
		if v, ok := any(item).(Validator); ok {
			if err := v.Validate(); err != nil {
				return fmt.Errorf("line %d: %w", lineNum, err)
			}
		}
		if err := fn(lineNum, item); err != nil {
			return err
		}
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("scan jsonl: %w", err)
	}
	return nil
}

// DecodeJSONLStrict decodes newline-delimited JSON records from r with DisallowUnknownFields
// by streaming through ForEachJSONL.
func DecodeJSONLStrict[T any](r io.Reader) ([]T, error) {
	var out []T
	err := ForEachJSONL[T](r, func(_ int, item T) error {
		out = append(out, item)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return out, nil
}

// DecodeYAMLStrict decodes YAML bytes with sigs.k8s.io/yaml.UnmarshalStrict
// and validates the decoded value via its Validate method.
func DecodeYAMLStrict[T Validator](data []byte) (*T, error) {
	var out T
	if err := yaml.UnmarshalStrict(data, &out); err != nil {
		return nil, fmt.Errorf("decode yaml: %w", err)
	}
	if err := out.Validate(); err != nil {
		return nil, err
	}
	return &out, nil
}
