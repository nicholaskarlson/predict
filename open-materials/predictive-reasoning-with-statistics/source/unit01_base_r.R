# Independent base-R joins, type-7 quantiles, empirical risks, and Brier scores.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("usage: Rscript unit01_base_r.R RESOURCE OUTDIR")
root <- args[1]; out <- args[2]
Sys.setlocale("LC_COLLATE", "C")
options(digits=17)
parts <- list()
for (cohort in c("anchor", "cohort2")) {
  read_input <- function(kind) {
    x <- read.csv(file.path(root,"source/tonmick-v1.0.0/data",cohort,paste0(kind,".csv")),
                  colClasses="character", na.strings=NULL, check.names=FALSE)
    if (anyDuplicated(x$student_id)) stop(paste("duplicate ID",kind))
    x
  }
  frame <- read_input("participants")
  survey <- read_input("baseline")
  registry <- read_input("registry")
  stopifnot(setequal(survey$student_id,registry$student_id),
            setequal(survey$student_id,frame$student_id[frame$responded_baseline=="1"]))
  # match performs key-based joins, with uniqueness checked before indexing.
  p <- frame[match(survey$student_id,frame$student_id),,drop=FALSE]
  r <- registry[match(survey$student_id,registry$student_id),,drop=FALSE]
  stopifnot(all(p$student_id==survey$student_id),all(r$student_id==survey$student_id),
            all(p$college_id==survey$college_id))
  selected <- if (cohort=="anchor") survey$offer=="0" else p$college_group=="continuing"
  p <- p[selected,,drop=FALSE]; r <- r[selected,,drop=FALSE]
  known <- r$exit_type!="consent_withdrawn"
  stopifnot(all(r$exit_type %in% c("none","withdrawal","transfer","consent_withdrawn")),
            all(as.integer(r$withdrawn_12[known])==as.integer(r$exit_type[known]=="withdrawal")),
            all(p$part_time %in% c("0","1")))
  grade <- as.numeric(p$prior_grade)
  stopifnot(all(is.finite(grade)),all(grade>=0 & grade<=4))
  parts[[cohort]] <- data.frame(row_id=p$student_id,
     role=ifelse(!known,"excluded_unknown",if (cohort=="anchor") "train" else "test"),
     part_time=as.integer(p$part_time),prior_grade=grade,
     y=ifelse(known,as.integer(r$withdrawn_12),NA_integer_),exit_type=r$exit_type,
     stringsAsFactors=FALSE)
}
rows <- do.call(rbind,parts)
stopifnot(!anyDuplicated(rows$row_id))
rows <- rows[order(rows$row_id),,drop=FALSE]
train <- rows[rows$role=="train",,drop=FALSE]
test <- rows[rows$role=="test",,drop=FALSE]
stopifnot(nrow(train)>0,nrow(test)>0)
cuts <- as.numeric(quantile(train$prior_grade,c(.25,.5,.75),type=7,names=FALSE))
baseline <- mean(train$y)
get_bin <- function(x) vapply(x,function(v) 1L+sum(v>cuts),integer(1))
train_bins <- get_bin(train$prior_grade)
cells <- data.frame()
for (part in 0:1) for (bin in 1:4) {
  selected <- train$part_time==part & train_bins==bin
  n <- sum(selected); events <- sum(train$y[selected])
  risk <- if (n>0) events/n else baseline
  cells <- rbind(cells,data.frame(part_time=part,grade_bin=bin,n=n,events=events,risk=risk,fallback=as.integer(n==0)))
}
pred <- rows[rows$role!="excluded_unknown",c("row_id","role","y","part_time"),drop=FALSE]
pred$grade_bin <- get_bin(rows$prior_grade[rows$role!="excluded_unknown"])
pred$table_probability <- cells$risk[4L*pred$part_time+pred$grade_bin]
pred$baseline_probability <- baseline
pred <- pred[,c("row_id","role","y","grade_bin","part_time","table_probability","baseline_probability")]
summary <- c(control_pool_n=nrow(train)+sum(rows$role=="excluded_unknown"),
             unknown_n=sum(rows$role=="excluded_unknown"),development_n=nrow(train),
             development_events=sum(train$y),temporal_n=nrow(test),temporal_events=sum(test$y),
             training_prevalence=baseline,temporal_prevalence=mean(test$y),
             development_transfer_n=sum(train$exit_type=="transfer"),temporal_transfer_n=sum(test$exit_type=="transfer"))
for (role in c("train","test")) {
  label <- if (role=="train") "development" else "temporal"
  selected <- pred[pred$role==role,,drop=FALSE]
  summary[paste0(label,"_brier_table")] <- mean((selected$table_probability-selected$y)^2)
  summary[paste0(label,"_brier_baseline")] <- mean((selected$baseline_probability-selected$y)^2)
}
summary["temporal_improvement"] <- summary["temporal_brier_baseline"]-summary["temporal_brier_table"]
summary["temporal_relative_improvement"] <- summary["temporal_improvement"]/summary["temporal_brier_baseline"]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
write_numeric <- function(x,name) {
  for (field in names(x)) if (is.numeric(x[[field]]))
    x[[field]] <- vapply(x[[field]],function(v) trimws(formatC(v,digits=17,format="g")),character(1))
  write.table(x,file.path(out,name),sep=",",quote=FALSE,row.names=FALSE,col.names=TRUE,na="")
}
write_numeric(data.frame(name=sort(names(summary)),value=unname(summary[sort(names(summary))])),"summary.csv")
write_numeric(data.frame(quantile=c(.25,.5,.75),value=cuts),"cutpoints.csv")
write_numeric(cells,"cells.csv")
write_numeric(pred,"predictions.csv")
write_numeric(data.frame(row_id=train$row_id),"fit_rows.csv")
writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))
cat("UNIT01_BASE_R_OK train=",nrow(train)," test=",nrow(test)," R=",as.character(getRversion()),"\n",sep="")
