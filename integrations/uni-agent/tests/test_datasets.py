# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Row shaping shared by every dataset source."""

from __future__ import annotations

import pytest

from uni_agent_substrate.datasets import (
    image_of,
    instance_id,
    r2e_row,
    select_rows,
    set_image,
)

EXAMPLE = {
    "repo_name": "numpy",
    "commit_hash": "abc123",
    "problem_statement": "Fix the thing.",
    "docker_image": "namanjain12/numpy_final:abc123",
    "expected_output_json": '{"test_a": "PASSED"}',
    "parsed_commit_content": "{}",
}


def test_r2e_row_shape_and_derived_instance_id():
    row = r2e_row(EXAMPLE, "some/dataset")
    task = row["extra_info"]["tools_kwargs"]["task"]
    assert row["data_source"] == "some/dataset"
    assert row["prompt"] == [{"role": "user", "content": "Fix the thing."}]
    assert task["name"] == "r2e_gym" and task["sandbox"]["image"] == EXAMPLE["docker_image"]
    assert task["metadata"]["expected_output_json"] == EXAMPLE["expected_output_json"]
    assert instance_id(row) == "numpy-abc123"
    assert image_of(row) == EXAMPLE["docker_image"]
    set_image(row, "registry.example/numpy_final@sha256:" + "a" * 64)
    assert image_of(row).startswith("registry.example/")


def test_r2e_row_requires_expected_output():
    with pytest.raises(ValueError):
        r2e_row({**EXAMPLE, "expected_output_json": ""}, "d")


def test_select_rows_sorted_head_and_seeded_shuffle():
    rows = [r2e_row({**EXAMPLE, "repo_name": repo, "commit_hash": str(i)}, "d") for i, repo in enumerate(["b"] * 5 + ["a"] * 5)]
    head = select_rows(rows, 3, None)
    assert [instance_id(r) for r in head] == ["a-5", "a-6", "a-7"]
    first = [instance_id(r) for r in select_rows(rows, 4, 0)]
    assert first == [instance_id(r) for r in select_rows(list(reversed(rows)), 4, 0)]  # input order does not matter
    assert first != [instance_id(r) for r in select_rows(rows, 4, 1)]
    assert len(select_rows(rows, None, 0)) == 10
