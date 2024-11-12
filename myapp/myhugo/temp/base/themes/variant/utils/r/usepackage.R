usepackage<-function(package){
  tryCatch(library(package), 
    error = function(e) install.packages(package, repos="https://cloud.r-project.org"),
    finally = library(package))
}