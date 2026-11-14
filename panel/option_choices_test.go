// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 jdominick05

package main

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestServeOptionsNumericChoices(t *testing.T) {
	// --cards is an argparse integer choice; --draft is a string choice. One numeric choice must not discard
	// the whole options response, including the advanced options that do not have choices.
	var opts []option
	err := json.Unmarshal([]byte(`[
		{"flag":"--cards","dest":"cards","type":"int","choices":[1,2]},
		{"flag":"--draft","dest":"draft","type":"str","choices":["off","mtp","dflash"]},
		{"flag":"--max-ctx","dest":"max_ctx","type":"ctx","default":"auto","choices":null}
	]`), &opts)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual([]string(opts[0].Choices), []string{"1", "2"}) || checkValue(opts[0], "2") != nil || checkValue(opts[0], "3") == nil {
		t.Fatalf("numeric choices must validate the entered argument: %+v", opts[0])
	}
	if checkValue(opts[1], "mtp") != nil || checkValue(opts[1], "other") == nil {
		t.Fatal("string choices must still constrain the entered argument")
	}
	m := newCtl(newCtlCLI("caminus"), true, "")
	next, _ := m.Update(optsMsg{v: opts})
	rows := strings.Join(next.(ctl).optionRows(20), "\n")
	if !strings.Contains(rows, "--max-ctx") || strings.Contains(rows, "gave no options") {
		t.Fatalf("advanced options disappeared:\n%s", rows)
	}
}

func TestServeOptionsInvalidChoices(t *testing.T) {
	for _, choice := range []string{"true", "null", "{}", "[]"} {
		var opts []option
		if err := json.Unmarshal([]byte(`[{"choices":[`+choice+`]}]`), &opts); err == nil {
			t.Errorf("non-argument choice accepted: %s", choice)
		}
	}
	var opts []option
	if err := json.Unmarshal([]byte(`[{"choices":[9007199254740993,"off"]}]`), &opts); err != nil {
		t.Fatal(err)
	}
	if opts[0].Choices[0] != "9007199254740993" {
		t.Fatal("numeric choices lost precision")
	}
}
