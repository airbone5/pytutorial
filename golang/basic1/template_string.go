// You can edit this code!
// Click here and start typing.
package main

import (
	"os"
	"text/template"
)

func main() {
	tmplString := "Hello,{{.}}"
	name := "fff xxx"
	tmpl := template.Must(template.New("hello").Parse(tmplString))
	err := tmpl.Execute(os.Stdout, name)
	if err != nil {
		panic("有問題")
	}

}
