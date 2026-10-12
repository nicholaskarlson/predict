# Independent R joins, item scoring, binomial IRLS, pair ranking and actions.
# The driver validates the frozen JSON contract; this route reads raw CSVs only.
read_chars <- function(path) read.csv(path,colClasses="character",na.strings=NULL,check.names=FALSE)
check <- function(condition,message) if (!isTRUE(condition)) stop(message,call.=FALSE)
item_score <- function(data,stem) {
  a <- matrix(NA_real_,nrow(data),5)
  for (j in 1:5) {
    v <- data[[paste0(stem,j)]]; present <- v!=""
    check(all(v[present] %in% as.character(1:5)),"invalid ordinal item")
    a[present,j] <- as.numeric(v[present])
    if (j==4) a[present,j] <- 6-a[present,j]
  }
  count <- rowSums(!is.na(a)); value <- rowSums(a,na.rm=TRUE)/count
  value[count<3] <- NA_real_
  list(value=value,count=count)
}
load_records <- function(root) {
  pieces <- list()
  for (cohort in c("anchor","cohort2")) {
    folder <- file.path(root,"source/tonmick-v1.0.0/data",cohort)
    raw <- lapply(c("participants","baseline","registry"),function(kind) {
      x <- read_chars(file.path(folder,paste0(kind,".csv")))
      check(!anyDuplicated(x$student_id),paste("duplicate raw key",kind)); x
    })
    frame <- raw[[1]]; survey <- raw[[2]]; registry <- raw[[3]]
    check(setequal(survey$student_id,registry$student_id) &&
          setequal(survey$student_id,frame$student_id[frame$responded_baseline=="1"]),"respondent membership")
    p <- frame[match(survey$student_id,frame$student_id),,drop=FALSE]
    r <- registry[match(survey$student_id,registry$student_id),,drop=FALSE]
    check(all(p$college_id==survey$college_id),"college mismatch")
    selected <- if (cohort=="anchor") rep(TRUE,nrow(p)) else p$college_group=="continuing"
    p <- p[selected,,drop=FALSE]; survey <- survey[selected,,drop=FALSE]; r <- r[selected,,drop=FALSE]
    check(all(r$exit_type %in% c("none","transfer","withdrawal","consent_withdrawn")),"invalid exit type")
    y <- ifelse(r$exit_type=="consent_withdrawn",NA_integer_,as.integer(r$exit_type=="withdrawal"))
    check(all(as.character(y[!is.na(y)])==r$withdrawn_12[!is.na(y)]),"outcome mismatch")
    wb <- item_score(survey,"wb"); st <- item_score(survey,"st")
    role <- ifelse(is.na(y),"excluded_unknown",ifelse(is.na(wb$value)|is.na(st$value),
             "excluded_unscorable_items",if (cohort=="anchor") "development" else "temporal_final"))
    check(all(p$part_time %in% c("0","1")) && all(p$first_gen %in% c("0","1")),"binary coding")
    check(all(p$gender %in% c("female","male","another")),"gender coding")
    check(all(p$program_area %in% c("Arts and Humanities","Social Sciences","STEM","Health","Business")),"program coding")
    age <- as.numeric(p$age); grade <- as.numeric(p$prior_grade)
    check(all(is.finite(age)&age>=16&age<=80) && all(is.finite(grade)&grade>=0&grade<=4),"invalid continuous predictor")
    offer <- if (cohort=="anchor") as.integer(survey$offer) else rep(0L,nrow(p))
    check(all(offer %in% 0:1),"offer coding")
    pieces[[cohort]] <- data.frame(row_id=p$student_id,cohort=cohort,college_id=p$college_id,
      region=p$region,part_time=as.integer(p$part_time),first_gen=as.integer(p$first_gen),age=age,
      gender=p$gender,program_area=p$program_area,prior_grade=grade,offer=offer,
      wellbeing_score=wb$value,stress_score=st$value,wellbeing_observed=wb$count,stress_observed=st$count,
      y=y,role=role,exit_type=r$exit_type,stringsAsFactors=FALSE)
  }
  rows <- do.call(rbind,pieces); rows <- rows[order(rows$row_id),,drop=FALSE]
  check(!anyDuplicated(rows$row_id),"overlapping cohort IDs")
  assignment <- read_chars(file.path(root,"splits/unit07_roles.csv"))
  for (key in c("row_id","cohort","college_id","role"))
    check(identical(as.character(rows[[key]]),assignment[[key]]),"assignment identity or role")
  rows
}
matrix_for <- function(data,model,processing,zero_offer=FALSE) {
  x <- matrix(1,nrow(data),1,dimnames=list(NULL,"intercept"))
  z <- function(key) (data[[key]]-processing[[key]][1])/processing[[key]][2]
  if (model=="entry_logistic") x <- cbind(x,z_prior_grade=z("prior_grade")) else {
    x <- cbind(x,part_time=data$part_time,first_gen=data$first_gen,z_age=z("age"),
      gender_male=as.integer(data$gender=="male"),gender_another=as.integer(data$gender=="another"))
    for (level in c("Social Sciences","STEM","Health","Business")) {
      x <- cbind(x,as.integer(data$program_area==level)); colnames(x)[ncol(x)] <- paste0("program_",gsub(" ","_",level))
    }
    x <- cbind(x,z_prior_grade=z("prior_grade"),z_wellbeing_score=z("wellbeing_score"),z_stress_score=z("stress_score"))
  }
  cbind(x,offer=if (zero_offer) rep(0,nrow(data)) else data$offer)
}
weighted_step <- function(x,y,p) {
  weights <- p*(1-p)
  check(all(is.finite(weights)&weights>0),"separation or nonpositive weight")
  q <- qr(x*sqrt(weights),tol=1e-10,LAPACK=FALSE)
  check(q$rank==ncol(x) && identical(q$pivot,seq_len(ncol(x))),"rank deficiency")
  triangular <- qr.R(q); d <- abs(diag(triangular))
  check(min(d)>1e-10*max(d),"rank deficiency")
  condition <- norm(triangular,"I")*norm(backsolve(triangular,diag(ncol(x))),"I")
  check(is.finite(condition) && condition<=1e5,"conditioning failure")
  list(step=as.numeric(qr.coef(q,(y-p)/sqrt(weights))),condition=condition,weights=weights)
}
fit_logistic <- function(x,y,maxit=100L) {
  check(nrow(x)>ncol(x) && length(y)==nrow(x) && all(is.finite(x)) && setequal(unique(y),0:1),"fit dimensions or classes")
  start <- c(log(sum(y)/(length(y)-sum(y))),rep(0,ncol(x)-1))
  weighted_step(x,y,plogis(as.numeric(x %*% start)))
  fit <- suppressWarnings(stats::glm.fit(x=x,y=y,family=stats::binomial(),start=start,
                control=stats::glm.control(epsilon=1e-12,maxit=maxit),singular.ok=FALSE))
  b <- as.numeric(fit$coefficients)
  check(isTRUE(fit$converged) && all(is.finite(b)) && max(abs(b))<=30,"separation or logistic nonconvergence")
  eta <- as.numeric(x %*% b); p <- plogis(eta); last <- weighted_step(x,y,p)
  score <- max(abs(as.numeric(crossprod(x,y-p)))); step <- max(abs(last$step))
  check(score<=1e-9 && step<=1e-9,"separation or logistic nonconvergence: score/step")
  signed <- ifelse(y==1,-eta,eta)
  nll <- sum(pmax(0,signed)+log1p(exp(-abs(signed))))
  list(beta=b,diagnostic=data.frame(n_fit=nrow(x),events=sum(y),p=ncol(x),rank=fit$rank,
    nll=nll,score_inf=score,newton_step_inf=step,condition_r_inf=last$condition,min_weight=min(last$weights),finite=1L))
}
score_probabilities <- function(y,p) {
  check(length(y)==length(p) && length(y)>0 && all(y %in% 0:1),"score population")
  check(all(is.finite(p)&p>=0&p<=1),"invalid probability")
  positive <- p[y==1]; negative <- p[y==0]
  comparison <- outer(positive,negative,"-")
  twice <- 2*sum(comparison>0)+sum(comparison==0); denominator <- 2*length(positive)*length(negative)
  q <- pmin(1-1e-12,pmax(1e-12,p))
  data.frame(n=length(y),events=length(positive),nonevents=length(negative),
    auc_twice_wins=if (denominator) twice else NA_real_,auc_denominator=denominator,
    auc=if (denominator) twice/denominator else NA_real_,auc_reason=if (denominator) "" else "one_class",
    brier=mean((p-y)^2),log_loss=-mean(ifelse(y==1,log(q),log1p(-q))),
    clipped_n=sum(p!=q),mean_probability=mean(p),stringsAsFactors=FALSE)
}
action_records <- function(predictions,model) {
  rank <- integer(nrow(predictions)); rank[order(-predictions$probability,predictions$row_id)] <- seq_len(nrow(predictions))
  summaries <- list(); manifest <- list(); n <- nrow(predictions)
  for (rule in c("threshold_020","capacity_300","all","none")) {
    selected <- switch(rule,threshold_020=predictions$probability>=.2,capacity_300=rank<=300,
                       all=rep(TRUE,n),none=rep(FALSE,n))
    tp <- sum(predictions$y[selected]); fp <- sum(selected)-tp
    summaries[[length(summaries)+1L]] <- data.frame(model_id=model,rule=rule,n=n,selected=sum(selected),
      tp=tp,fp=fp,tn=n-sum(predictions$y)-fp,fn=sum(predictions$y)-tp,
      net_benefit=(tp-fp/4)/n,utility_total=4*tp-fp)
    if (rule %in% c("threshold_020","capacity_300")) manifest[[length(manifest)+1L]] <-
      data.frame(model_id=model,rule=rule,row_id=predictions$row_id,rank=rank,selected=as.integer(selected))
  }
  list(decisions=do.call(rbind,summaries),action_ids=do.call(rbind,manifest),rank=rank)
}
calculate <- function(root) {
  rows <- load_records(root); training <- rows[rows$role=="development",,drop=FALSE]
  later <- rows[rows$role=="temporal_final",,drop=FALSE]; controls <- training[training$offer==0,,drop=FALSE]
  check(nrow(controls)>0 && nrow(later)>=300,"population or capacity")
  prevalence <- mean(controls$y); cuts <- as.numeric(quantile(controls$prior_grade,c(.25,.5,.75),type=7,names=FALSE))
  grade_bin <- function(v) vapply(v,function(a) sum(a>cuts)+1L,integer(1))
  cells <- list()
  for (part in 0:1) for (bin in 1:4) {
    group <- controls[controls$part_time==part & grade_bin(controls$prior_grade)==bin,,drop=FALSE]
    cells[[length(cells)+1L]] <- data.frame(part_time=part,grade_bin=bin,n=nrow(group),events=sum(group$y),
         probability=if (nrow(group)) mean(group$y) else prevalence,fallback=as.integer(nrow(group)==0))
  }
  outputs <- setNames(lapply(1:9,function(i) list()),c("predictions","coefficients","diagnostics","transforms",
                      "fit_rows","scores","decisions","action_ids","examples"))
  add <- function(name,value) outputs[[name]][[length(outputs[[name]])+1L]] <<- value
  table <- do.call(rbind,cells)
  for (model in c("control_mean","control_table","entry_logistic","full_logistic")) {
    membership <- if (model %in% c("control_mean","control_table")) controls else training
    add("fit_rows",data.frame(model_id=model,row_id=membership$row_id))
    if (model %in% c("control_mean","control_table")) {
      p <- if (model=="control_mean") rep(prevalence,nrow(later)) else table$probability[4*later$part_time+grade_bin(later$prior_grade)]
      eta <- qlogis(p)
    } else {
      keys <- if (model=="entry_logistic") "prior_grade" else c("age","prior_grade","wellbeing_score","stress_score")
      processing <- setNames(lapply(keys,function(key) {
        value <- training[[key]]; m <- mean(value); s <- sqrt(mean((value-m)^2))
        check(is.finite(s) && s>0,"zero or invalid training SD"); c(m,s)
      }),keys)
      for (key in keys) add("transforms",data.frame(model_id=model,feature=key,n_fit=nrow(training),mean=processing[[key]][1],sd=processing[[key]][2]))
      x <- matrix_for(training,model,processing); fit <- fit_logistic(x,training$y)
      add("coefficients",data.frame(model_id=model,term=colnames(x),coefficient=fit$beta))
      add("diagnostics",data.frame(model_id=model,fit$diagnostic))
      eta <- as.numeric(matrix_for(later,model,processing,TRUE) %*% fit$beta); p <- plogis(eta)
    }
    q <- pmin(1-1e-12,pmax(1e-12,p))
    prediction <- data.frame(model_id=model,row_id=later$row_id,y=later$y,offer_for_prediction=0L,
      probability=p,eta=eta,brier_loss=(p-later$y)^2,log_loss=ifelse(later$y==1,-log(q),-log1p(-q)))
    add("predictions",prediction); add("scores",data.frame(model_id=model,score_probabilities(later$y,p)))
    action <- action_records(prediction,model); add("decisions",action$decisions); add("action_ids",action$action_ids)
    if (model=="full_logistic") add("examples",data.frame(row_id=later$row_id[1:3],y=later$y[1:3],
      wellbeing_score=later$wellbeing_score[1:3],stress_score=later$stress_score[1:3],eta=eta[1:3],probability=p[1:3],
      brier_loss=prediction$brier_loss[1:3],log_loss=prediction$log_loss[1:3],rank=action$rank[1:3],
      threshold_selected=as.integer(p[1:3]>=.2),capacity_selected=as.integer(action$rank[1:3]<=300)))
  }
  outputs <- lapply(outputs,function(x) do.call(rbind,x))
  outputs$table_cells <- table; outputs$cutpoints <- data.frame(quantile=c(.25,.5,.75),value=cuts)
  summary <- c(anchor_pool_n=sum(rows$cohort=="anchor"),anchor_unknown_n=sum(rows$cohort=="anchor" & rows$role=="excluded_unknown"),
    anchor_unscorable_n=sum(rows$cohort=="anchor" & rows$role=="excluded_unscorable_items"),development_n=nrow(training),
    development_events=sum(training$y),development_offer0_n=nrow(controls),development_offer0_events=sum(controls$y),
    development_offer1_n=sum(training$offer==1),development_offer1_events=sum(training$y[training$offer==1]),
    temporal_n=nrow(later),temporal_events=sum(later$y),temporal_unscorable_n=sum(rows$cohort=="cohort2" & rows$role=="excluded_unscorable_items"),
    development_transfer_n=sum(training$exit_type=="transfer"),temporal_transfer_n=sum(later$exit_type=="transfer"),
    candidate_parameters_n=13,full_coefficients_n=14,events_per_candidate_parameter=sum(training$y)/13,
    control_prevalence=prevalence,temporal_prevalence=mean(later$y),
    development_partial_wb_n=sum(training$wellbeing_observed<5),development_partial_st_n=sum(training$stress_observed<5),
    temporal_partial_wb_n=sum(later$wellbeing_observed<5),temporal_partial_st_n=sum(later$stress_observed<5))
  outputs$summary <- data.frame(name=sort(names(summary)),value=unname(summary[sort(names(summary))]))
  outputs
}
emit <- function(outputs,out) {
  dir.create(out,recursive=TRUE,showWarnings=FALSE)
  for (name in names(outputs)) {
    data <- outputs[[name]]
    for (key in names(data)) if (is.numeric(data[[key]])) data[[key]] <-
      vapply(data[[key]],function(v) if (is.na(v)) "" else sprintf("%.17g",v),character(1))
    write.table(data,file.path(out,paste0(name,".csv")),sep=",",row.names=FALSE,col.names=TRUE,quote=FALSE,na="")
  }
  writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))
}
if (sys.nframe()==0L) {
  args <- commandArgs(trailingOnly=TRUE)
  if (length(args)!=2L) stop("usage: Rscript unit07_base_r.R RESOURCE OUTDIR")
  Sys.setlocale("LC_COLLATE","C"); options(digits=17)
  result <- calculate(args[1]); emit(result,args[2])
  cat("UNIT07_BASE_R_OK logistic_fits=2 predictions=",nrow(result$predictions)," R=",as.character(getRversion())," solver=glm.fit_IRLS_QR\n",sep="")
}
