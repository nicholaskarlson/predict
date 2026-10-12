# Independent base-R key joins, fold-specific processing, LINPACK QR and scores.
# The Python orchestrator checks the frozen model record before this route runs.
# No analyzer output, prediction artifact or Python fitting helper is read here.
args <- commandArgs(trailingOnly=TRUE)
if (length(args)!=2L) stop("usage: Rscript unit04_base_r.R RESOURCE OUTDIR")
root <- args[1]; out <- args[2]
Sys.setlocale("LC_COLLATE","C"); options(digits=17)
read_chars <- function(path) read.csv(path,colClasses="character",na.strings=NULL,check.names=FALSE)
pieces <- list()
for (cohort in c("anchor","cohort2")) {
  folder <- file.path(root,"source/tonmick-v1.0.0/data",cohort)
  raw <- lapply(c("participants","baseline","registry"),function(kind) {
    x <- read_chars(file.path(folder,paste0(kind,".csv")))
    if (anyDuplicated(x$student_id)) stop(paste("duplicate raw key",kind))
    x
  })
  frame <- raw[[1]]; survey <- raw[[2]]; registry <- raw[[3]]
  stopifnot(setequal(survey$student_id,registry$student_id),
            setequal(survey$student_id,frame$student_id[frame$responded_baseline=="1"]))
  p <- frame[match(survey$student_id,frame$student_id),,drop=FALSE]
  r <- registry[match(survey$student_id,registry$student_id),,drop=FALSE]
  stopifnot(all(p$student_id==survey$student_id),all(r$student_id==survey$student_id),all(p$college_id==survey$college_id))
  selected <- if (cohort=="anchor") survey$offer=="0" else p$college_group=="continuing"
  p <- p[selected,,drop=FALSE]; r <- r[selected,,drop=FALSE]
  y <- rep(NA_real_,nrow(r)); observed <- r$gpa12!=""; y[observed] <- as.numeric(r$gpa12[observed])
  stopifnot(all(is.finite(y[observed])),all(y[observed]>=0 & y[observed]<=4),
            all(!observed[r$exit_type=="consent_withdrawn"]),
            all(p$part_time %in% c("0","1")),all(p$first_gen %in% c("0","1")))
  age <- as.numeric(p$age); prior <- as.numeric(p$prior_grade)
  stopifnot(all(is.finite(age)),all(age>=16 & age<=80),all(is.finite(prior)),all(prior>=0 & prior<=4))
  pieces[[cohort]] <- data.frame(row_id=p$student_id,cohort=cohort,college_id=p$college_id,region=p$region,
    part_time=as.integer(p$part_time),first_gen=as.integer(p$first_gen),age=age,prior_grade=prior,offer=0L,
    exit_type=r$exit_type,exit_month=as.numeric(r$exit_month),gpa_at_exit=as.integer(r$gpa_at_exit),y=y,
    role=ifelse(!observed,"excluded_missing_grade",if (cohort=="anchor") "development" else "temporal_final"),
    stringsAsFactors=FALSE)
}
rows <- do.call(rbind,pieces); rows <- rows[order(rows$row_id),,drop=FALSE]
stopifnot(!anyDuplicated(rows$row_id))
assignments <- read_chars(file.path(root,"splits/unit04_assignments.csv"))
stopifnot(identical(rows$row_id,assignments$row_id))
for (key in c("cohort","college_id","region","role")) stopifnot(identical(as.character(rows[[key]]),assignments[[key]]))
development <- rows$role=="development"
stopifnot(all(assignments$student_fold[development] %in% as.character(1:5)),
          all(assignments$region_fold[development]==rows$region[development]),
          all(assignments$college_fold[development]!=""))
for (key in c("student_fold","college_fold","region_fold")) {
  stopifnot(all(assignments[[key]][!development]=="")); rows[[key]] <- assignments[[key]]
}
college_map <- split(rows$college_fold[development],rows$college_id[development])
stopifnot(all(vapply(college_map,function(x) length(unique(x))==1L,logical(1))),
          length(unique(vapply(college_map,function(x) x[1],character(1))))==length(college_map))
train_all <- rows[development,,drop=FALSE]; jobs <- list()
for (design in c("student","college","region")) {
  key <- paste0(design,"_fold")
  for (fold in sort(unique(train_all[[key]]))) {
    fit <- train_all[train_all[[key]]!=fold,,drop=FALSE]
    held <- train_all[train_all[[key]]==fold,,drop=FALSE]
    stopifnot(nrow(fit)>0,nrow(held)>0)
    if (design!="student") {
      group <- if (design=="college") "college_id" else "region"
      stopifnot(!any(unique(held[[group]]) %in% unique(fit[[group]])))
    }
    jobs[[length(jobs)+1L]] <- list(design=design,fold=fold,fit=fit,held=held)
  }
}
jobs[[length(jobs)+1L]] <- list(design="temporal",fold="final",fit=train_all,
                              held=rows[rows$role=="temporal_final",,drop=FALSE])

matrix_for <- function(data,model,processing,colleges) {
  x <- matrix(1,nrow(data),1,dimnames=list(NULL,"intercept"))
  if (model!="mean") x <- cbind(x,z_prior_grade=(data$prior_grade-processing$prior_grade[1])/processing$prior_grade[2])
  if (model %in% c("baseline","college"))
    x <- cbind(x,part_time=data$part_time,first_gen=data$first_gen,z_age=(data$age-processing$age[1])/processing$age[2])
  if (model=="college") for (college in head(colleges,-1)) {
    effect <- ifelse(data$college_id==college,1,ifelse(data$college_id==tail(colleges,1),-1,0))
    x <- cbind(x,effect); colnames(x)[ncol(x)] <- paste0("effect_",college)
  }
  x
}

outputs <- setNames(lapply(1:7,function(i) list()),c("predictions","coefficients","diagnostics","transforms","fit_rows","fold_scores","aggregate"))
append_output <- function(name,value) { outputs[[name]][[length(outputs[[name]])+1L]] <<- value }
for (job in jobs) {
  fit <- job$fit; held <- job$held; design <- job$design; fold <- job$fold
  stopifnot(nrow(held)>0)
  processing <- lapply(c("prior_grade","age"),function(key) {
    values <- fit[[key]]; m <- mean(values); sd <- sqrt(mean((values-m)^2))
    stopifnot(is.finite(sd),sd>1e-12); c(m,sd)
  }); names(processing) <- c("prior_grade","age")
  colleges <- sort(unique(fit$college_id))
  append_output("fit_rows",data.frame(design=design,fold=fold,row_id=fit$row_id))
  for (key in names(processing)) append_output("transforms",data.frame(design=design,fold=fold,column=key,
                               mean=processing[[key]][1],sd=processing[[key]][2]))
  for (model in c("mean","entry","baseline","college")) {
    x <- matrix_for(fit,model,processing,colleges); v <- matrix_for(held,model,processing,colleges)
    stopifnot(nrow(x)>=ncol(x))
    decomp <- qr(x,tol=1e-10,LAPACK=FALSE)
    stopifnot(decomp$rank==ncol(x),identical(decomp$pivot,seq_len(ncol(x))))
    triangular <- qr.R(decomp); diagonal <- abs(diag(triangular))
    stopifnot(all(diagonal>1e-10*max(diagonal)))
    condition <- norm(triangular,"I")*norm(backsolve(triangular,diag(ncol(x))),"I")
    stopifnot(is.finite(condition),condition<=1e5)
    beta <- as.numeric(qr.coef(decomp,fit$y)); fitted <- as.numeric(x %*% beta)
    score <- max(abs(as.numeric(crossprod(x,fit$y-fitted))))/nrow(x)
    stopifnot(all(is.finite(beta)),score<=1e-10)
    append_output("coefficients",data.frame(design=design,fold=fold,model_id=model,term=colnames(x),coefficient=beta))
    append_output("diagnostics",data.frame(design=design,fold=fold,model_id=model,n_fit=nrow(fit),p=ncol(x),rank=decomp$rank,
                    condition_r_inf=condition,score_inf=score,fit_mse=mean((fit$y-fitted)^2)))
    forecast <- as.numeric(v %*% beta); error <- held$y-forecast
    append_output("predictions",data.frame(design=design,fold=fold,model_id=model,row_id=held$row_id,y=held$y,
                    prediction=forecast,squared_error=error^2,absolute_error=abs(error),
                    unseen_college=as.integer(!held$college_id %in% colleges)))
    append_output("fold_scores",data.frame(design=design,fold=fold,model_id=model,n=nrow(held),
                    n_unseen=sum(!held$college_id %in% colleges),mse=mean(error^2),mae=mean(abs(error)),rmse=sqrt(mean(error^2))))
  }
}
for (name in names(outputs)[names(outputs)!="aggregate"]) outputs[[name]] <- do.call(rbind,outputs[[name]])
for (design in c("student","college","region","temporal")) for (model in c("mean","entry","baseline","college")) {
  selected <- outputs$predictions[outputs$predictions$design==design & outputs$predictions$model_id==model,,drop=FALSE]
  folds <- outputs$fold_scores$mse[outputs$fold_scores$design==design & outputs$fold_scores$model_id==model]
  mse <- mean(selected$squared_error)
  append_output("aggregate",data.frame(design=design,model_id=model,n=nrow(selected),folds=length(folds),mse=mse,
    mae=mean(selected$absolute_error),rmse=sqrt(mse),fold_mean_mse=mean(folds),fold_min_mse=min(folds),fold_max_mse=max(folds)))
}
outputs$aggregate <- do.call(rbind,outputs$aggregate)
summary <- c()
for (cohort in c("anchor","cohort2")) {
  label <- if (cohort=="anchor") "anchor" else "temporal"
  pool <- rows[rows$cohort==cohort,,drop=FALSE]; eligible <- pool[!is.na(pool$y),,drop=FALSE]
  values <- c(pool_n=nrow(pool),grade_n=nrow(eligible),missing_n=nrow(pool)-nrow(eligible),
              consent_missing_n=sum(pool$exit_type=="consent_withdrawn"),exit_grade_n=sum(eligible$gpa_at_exit),
              college_n=length(unique(eligible$college_id)),grade_mean=mean(eligible$y))
  names(values) <- paste0(label,"_",names(values)); summary <- c(summary,values)
}
summary["wrong_zero_anchor_mean"] <- sum(train_all$y)/summary["anchor_pool_n"]
examples <- outputs$predictions[outputs$predictions$design=="temporal" & outputs$predictions$model_id=="baseline",,drop=FALSE]
examples <- examples[order(examples$row_id),,drop=FALSE][1:3,,drop=FALSE]
summary["example_mse"] <- mean(examples$squared_error)
summary["example_mae"] <- mean(examples$absolute_error)
summary["example_rmse"] <- sqrt(summary["example_mse"])
outputs$examples <- examples
outputs$summary <- data.frame(name=sort(names(summary)),value=unname(summary[sort(names(summary))]))
dir.create(out,recursive=TRUE,showWarnings=FALSE)
numeric_fields <- c("y","prediction","squared_error","absolute_error","coefficient","n_fit","p","rank","condition_r_inf",
                    "score_inf","fit_mse","mean","sd","n","n_unseen","mse","mae","rmse","folds","fold_mean_mse","fold_min_mse","fold_max_mse","value")
for (name in names(outputs)) {
  data <- outputs[[name]]; keys <- setdiff(names(data),numeric_fields)
  if (length(keys)>0) data <- data[do.call(order,lapply(data[keys],as.character)),,drop=FALSE]
  for (key in names(data)) if (is.numeric(data[[key]]))
    data[[key]] <- vapply(data[[key]],function(v) sprintf("%.17g",v),character(1))
  write.table(data,file.path(out,paste0(name,".csv")),sep=",",row.names=FALSE,col.names=TRUE,quote=FALSE,na="")
}
writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))
cat("UNIT04_BASE_R_OK fits=",nrow(outputs$diagnostics)," predictions=",nrow(outputs$predictions),
    " R=",as.character(getRversion())," solver=LINPACK_QR\n",sep="")
