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
    selected <- if (cohort=="anchor") survey$offer=="0" else rep(TRUE,nrow(p))
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
      y=y,role=role,exit_type=r$exit_type,college_group=if(cohort=="anchor") "anchor" else p$college_group,grade=ifelse(r$gpa12=="",NA_real_,as.numeric(r$gpa12)),gpa_at_exit=as.integer(r$gpa_at_exit),stringsAsFactors=FALSE)
  }
  rows <- do.call(rbind,pieces); rows <- rows[order(rows$row_id),,drop=FALSE]
  check(!anyDuplicated(rows$row_id),"overlapping cohort IDs")
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

fit_cal <- function(eta,y,kind) {
 x <- if(kind=="offset") matrix(1,length(y),1) else cbind(1,eta)
 offset <- if(kind=="offset") eta else rep(0,length(y))
 check(length(y)>ncol(x) && all(is.finite(x)) && setequal(unique(y),0:1),"calibration classes or dimensions")
 start <- c(log(sum(y)/(length(y)-sum(y))),rep(0,ncol(x)-1))
 fit <- suppressWarnings(glm.fit(x,y,family=binomial(),offset=offset,start=start,control=glm.control(epsilon=1e-13,maxit=200),singular.ok=FALSE))
 b <- as.numeric(fit$coefficients); check(isTRUE(fit$converged),"nonconvergence")
 for(i in 0:200) {
  check(all(is.finite(b)) && max(abs(b))<=30,"separation or coefficient boundary")
  lp <- as.numeric(x %*% b)+offset;p <- plogis(lp);step <- weighted_step(x,y,p);score <- max(abs(crossprod(x,y-p)))
  if(score<=1e-10 && max(abs(step$step))<=1e-10) {
   z <- ifelse(y==1,-lp,lp);nll <- sum(pmax(0,z)+log1p(exp(-abs(z))))
   return(list(b=b,d=data.frame(n_fit=length(y),events=sum(y),p=ncol(x),rank=ncol(x),nll=nll,score_inf=score,newton_step_inf=max(abs(step$step)),condition_r_inf=step$condition,min_weight=min(step$weights),finite=1)))
  }
  b <- b+step$step
 }
 stop("nonconvergence")
}
conformal_quantile <- function(scores,numerator=1L,denominator=10L) {
 check(length(scores)>0 && all(is.finite(scores)&scores>=0),"invalid residuals")
 check(numerator>0 && numerator<denominator,"invalid alpha")
 n <- length(scores);k <- ceiling((n+1)*(denominator-numerator)/denominator)
 list(k=k,q=if(k>n) Inf else sort(scores)[k])
}
logits <- function(p) {check(all(is.finite(p)&p>=0&p<=1),"probability range");q <- pmin(1-1e-12,pmax(1e-12,p));log(q)-log1p(-q)}
diagnose <- function(eta,y,kind) {
 if(length(unique(y))<2) return(list(fit=NULL,reason="one_class"))
 if(kind=="joint" && diff(range(eta))<=1e-12) return(list(fit=NULL,reason="constant_prediction"))
 tryCatch(list(fit=fit_cal(eta,y,kind),reason=""),error=function(e) list(fit=NULL,reason=paste0("fit_failure:",conditionMessage(e))))
}
calculate <- function(root) {
 rows <- load_records(root);a <- read_chars(file.path(root,"splits/unit08_roles.csv"));check(nrow(a)==nrow(rows) && !anyDuplicated(a$row_id),"manifest uniqueness and coverage")
 for(key in c("row_id","cohort","college_id")) check(identical(as.character(rows[[key]]),a[[key]]),"manifest keys")
 rows$assignment <- a$assignment
 check(all(ifelse(rows$college_group=="anchor",rows$assignment %in% c("grade_fit","grade_calibration"),ifelse(rows$college_group=="new",rows$assignment %in% c("new_updating","new_final"),rows$assignment=="continuing_final"))),"invalid population role")
 check(all(is.na(rows$grade)|is.finite(rows$grade)&rows$grade>=0&rows$grade<=4&rows$exit_type!="consent_withdrawn"),"invalid grade")
 risk <- rows[rows$cohort=="cohort2" & !is.na(rows$y)&!is.na(rows$wellbeing_score)&!is.na(rows$stress_score),,drop=FALSE]
 co <- read_chars(file.path(root,"expected-results/unit07_coefficients.csv"));co <- co[co$model_id=="full_logistic",];tr <- read_chars(file.path(root,"expected-results/unit07_transforms.csv"));tr <- tr[tr$model_id=="full_logistic",];processing <- lapply(seq_len(nrow(tr)),function(j) c(as.numeric(tr$mean[j]),as.numeric(tr$sd[j])));names(processing) <- tr$feature
 risk$p <- plogis(as.numeric(matrix_for(risk,"full_logistic",processing,TRUE) %*% as.numeric(co$coefficient)));risk$eta <- logits(risk$p)
 result <- setNames(rep(list(NULL),11),c("predictions","binary_scores","bins","coefficients","diagnostics","fit_rows","residuals","intervals","coverage","selection","summary"))
 append <- function(key,frame) result[[key]] <<- rbind(result[[key]],frame)
 register <- function(id,scope,rows,fit) {
  append("coefficients",data.frame(fit_id=id,scope=scope,term=c("intercept","slope")[seq_along(fit$b)],coefficient=fit$b));append("diagnostics",data.frame(fit_id=id,scope=scope,fit$d));append("fit_rows",data.frame(fit_id=id,scope=scope,row_id=rows$row_id))
 }
 update <- risk[risk$assignment=="new_updating",];fits <- list()
 for(kind in c("offset","joint")) {f <- fit_cal(update$eta,update$y,kind);fits[[kind]] <- f$b;register(paste0("update_",kind),"updating_rule",update,f)}
 for(role in c("continuing_final","new_final")) {
  test <- risk[risk$assignment==role,];y <- test$y
  methods <- if(role=="continuing_final") "unchanged" else c("unchanged","offset","joint")
  for(method in methods) {
   p <- if(method=="unchanged") test$p else plogis(fits[[method]][1]+(if(method=="offset") 1 else fits[[method]][2])*test$eta)
   eta <- logits(p);cal <- list()
   for(kind in c("offset","joint")) {cal[[kind]] <- diagnose(eta,y,kind);if(!is.null(cal[[kind]]$fit)) register(paste0("diagnostic_",role,"_",method,"_",kind),"diagnostic_only",test,cal[[kind]]$fit)}
   q <- pmin(1-1e-12,pmax(1e-12,p));append("predictions",data.frame(role=role,method=method,row_id=test$row_id,y=y,probability=p))
   append("binary_scores",data.frame(role=role,method=method,n=length(y),events=sum(y),observed_fraction=mean(y),mean_probability=mean(p),observed_minus_predicted=mean(y)-mean(p),brier=mean((y-p)^2),log_loss=mean(ifelse(y==1,-log(q),-log1p(-q))),offset_intercept=if(is.null(cal$offset$fit)) NA else cal$offset$fit$b[1],offset_reason=cal$offset$reason,joint_intercept=if(is.null(cal$joint$fit)) NA else cal$joint$fit$b[1],joint_slope=if(is.null(cal$joint$fit)) NA else cal$joint$fit$b[2],joint_reason=cal$joint$reason))
   for(j in 0:4) {ids <- which(p>=c(0,.2,.4,.6,.8,1)[j+1] & (p<c(0,.2,.4,.6,.8,1)[j+2]|j==4&p<=1));append("bins",data.frame(role=role,method=method,bin=j+1,n=length(ids),mean_probability=if(length(ids)) mean(p[ids]) else NA,observed_fraction=if(length(ids)) mean(y[ids]) else NA,reason=if(length(ids)) "" else "empty_bin"))}
  }
 }
 fit <- rows[rows$assignment=="grade_fit"&!is.na(rows$grade),];cal <- rows[rows$assignment=="grade_calibration"&!is.na(rows$grade),];x <- cbind(1,fit$prior_grade);qrfit <- qr(x,tol=1e-10);check(qrfit$rank==2,"grade rank");rr <- qr.R(qrfit);condition <- norm(rr,"I")*norm(backsolve(rr,diag(2)),"I");check(condition<=1e5,"grade condition");beta <- as.numeric(qr.coef(qrfit,fit$grade));pred <- function(data) beta[1]+beta[2]*data$prior_grade
 append("coefficients",data.frame(fit_id="grade_ols",scope="grade_rule",term=c("intercept","slope"),coefficient=beta));append("fit_rows",data.frame(fit_id="grade_ols",scope="grade_rule",row_id=fit$row_id));append("fit_rows",data.frame(fit_id="grade_quantile",scope="interval_calibration",row_id=cal$row_id))
 res <- data.frame(row_id=cal$row_id,prediction=pred(cal),y=cal$grade,absolute_residual=abs(cal$grade-pred(cal)));res <- res[order(res$absolute_residual,res$row_id),];append("residuals",data.frame(rank=seq_len(nrow(res)),res));chosen <- conformal_quantile(res$absolute_residual);k <- chosen$k;q <- chosen$q;append("selection",data.frame(n_fit=nrow(fit),n_calibration=nrow(cal),k=k,q=q,width=2*q,condition_r_inf=condition))
 for(role in c("continuing_final","new_final")) {
  test <- rows[rows$assignment==role&!is.na(rows$grade),];p <- pred(test);records <- data.frame(role=role,row_id=test$row_id,part_time=test$part_time,y=test$grade,prediction=p,lower=p-q,upper=p+q,covered=as.integer(p-q<=test$grade&test$grade<=p+q),width=2*q,gpa_at_exit=test$gpa_at_exit);append("intervals",records)
  for(group in c("all","part_time_0","part_time_1")) {a <- if(group=="all") records else records[records$part_time==as.integer(substr(group,11,11)),];append("coverage",data.frame(role=role,group=group,n=nrow(a),covered=sum(a$covered),coverage=mean(a$covered),mean_width=mean(a$width),exit_grades=sum(a$gpa_at_exit),bounds_outside_0_4=sum(a$lower<0|a$upper>4)))}
 }
 for(role in c("grade_fit","grade_calibration","continuing_final","new_updating","new_final")) {a <- rows[rows$assignment==role,];b <- risk[risk$assignment==role,];append("summary",data.frame(role=role,pool_n=nrow(a),binary_n=nrow(b),unknown_n=sum(is.na(a$y)),unscorable_known_n=sum(!is.na(a$y)&(is.na(a$wellbeing_score)|is.na(a$stress_score))),observed_grade_n=sum(!is.na(a$grade)),missing_grade_n=sum(is.na(a$grade)),colleges=length(unique(a$college_id))))}
 result
}
args <- commandArgs(trailingOnly=TRUE)
if(length(args)) {
 check(length(args)==2,"usage root outdir");root <- args[1];out <- args[2]
 # Scientific driver and this independent route both protect the frozen JSON.
 digest <- strsplit(system2("sha256sum",shQuote(file.path(root,"model-specs/unit08.json")),stdout=TRUE)," ")[[1]][1]
 check(identical(digest,"4285c24ae71294208bab1c575adfcc2b4e9a91cdbe5787cfa3d8f0c4311d5798"),"frozen specification altered")
 options(digits=17);result <- calculate(root);dir.create(out,recursive=TRUE,showWarnings=FALSE)
 for(key in names(result)) {frame <- result[[key]];for(j in seq_along(frame)) if(is.numeric(frame[[j]])) frame[[j]] <- ifelse(is.na(frame[[j]]),"",sprintf("%.17g",frame[[j]]));write.table(frame,file.path(out,paste0(key,".csv")),sep=",",quote=FALSE,row.names=FALSE,na="")}
 cat("UNIT08_BASE_R_OK predictions=",nrow(result$predictions)," intervals=",nrow(result$intervals)," R=",as.character(getRversion()),"\n",sep="")
}
