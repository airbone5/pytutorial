package main

import (
	"html/template"
	"os"
	"strings"
)

func main() {
	tmplString := "Hello, {{.}}"
	name := "Feury"
	tmpl := template.Must(template.New("hello").Parse(tmplString))
	err := tmpl.Execute(os.Stdout, name)
	if err != nil {
		panic("Could not execute template")
	}

	tmplString = "{{ if . }}Hello{{ end }}"
	tmpl = template.Must(template.New("hello2").Parse(tmplString))
	_ = tmpl.Execute(os.Stdout, 1)
	names := []string{"Donald", "Bob", "Brodie"}
	tmplString = "{{ range . }}Hello, {{.}}\n{{ end }}"
	tmpl = template.Must(template.New("range").Parse(tmplString))
	_ = tmpl.Execute(os.Stdout, names)
	tmplString = "{{ range . }}Hello, {{ toUpper . }}\n{{ end }}"
	funcMap := map[string]interface{}{
		"toUpper": strings.ToUpper,
	}
	tmpl = template.Must(template.New("funcs").Funcs(funcMap).Parse(tmplString))
	_ = tmpl.Execute(os.Stdout, names)
}
