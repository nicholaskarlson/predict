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

fit_general <- function(x,y,offset=rep(0,length(y))) {
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
fit_cal <- function(eta,y,kind) fit_general(if(kind=="offset") matrix(1,length(y),1) else cbind(1,eta),y,if(kind=="offset") eta else rep(0,length(y)))
logits <- function(p) {check(all(is.finite(p)&p>=0&p<=1),"probability range");q <- pmin(1-1e-12,pmax(1e-12,p));log(q)-log1p(-q)}
diagnose <- function(eta,y,kind) {
 if(length(unique(y))<2) return(list(fit=NULL,reason="one_class"))
 if(kind=="joint" && diff(range(eta))<=1e-12) return(list(fit=NULL,reason="constant_prediction"))
 tryCatch(list(fit=fit_cal(eta,y,kind),reason=""),error=function(e) list(fit=NULL,reason=paste0("fit_failure:",conditionMessage(e))))
}

auc_score <- function(y,p) {
 n1 <- sum(y);n0 <- length(y)-n1
 if(!n1 || !n0) return(list(value=NA_real_,reason="one_class"))
 ranks <- rank(p,ties.method="average");list(value=(sum(ranks[y==1])-n1*(n1+1)/2)/(n1*n0),reason="")
}
score <- function(y,p) {
 check(length(y)>0 && length(y)==length(p) && all(y %in% 0:1),"binary labels");eta <- logits(p);q <- pmin(1-1e-12,pmax(1e-12,p));cal <- lapply(c("offset","joint"),function(k) diagnose(eta,y,k));names(cal) <- c("offset","joint");auc <- auc_score(y,p)
 list(s=data.frame(n=length(y),events=sum(y),auc=auc$value,auc_reason=auc$reason,observed_fraction=mean(y),mean_probability=mean(p),brier=mean((y-p)^2),log_loss=mean(ifelse(y==1,-log(q),-log1p(-q))),clipped_n=sum(p!=q),offset_intercept=if(is.null(cal$offset$fit)) NA else cal$offset$fit$b[1],offset_reason=cal$offset$reason,joint_intercept=if(is.null(cal$joint$fit)) NA else cal$joint$fit$b[1],joint_slope=if(is.null(cal$joint$fit)) NA else cal$joint$fit$b[2],joint_reason=cal$joint$reason),cal=cal)
}
composition <- function(draw,sites,method,base="unchanged") {
 n <- sum(vapply(draw,function(k) sites[[k]][[method]]$n,numeric(1)))
 b <- sum(vapply(draw,function(k) sites[[k]][[method]]$brier*sites[[k]][[method]]$n,numeric(1)))/n
 l <- sum(vapply(draw,function(k) sites[[k]][[method]]$log_loss*sites[[k]][[method]]$n,numeric(1)))/n
 b0 <- sum(vapply(draw,function(k) sites[[k]][[base]]$brier*sites[[k]][[base]]$n,numeric(1)))/n
 l0 <- sum(vapply(draw,function(k) sites[[k]][[base]]$log_loss*sites[[k]][[base]]$n,numeric(1)))/n
 data.frame(n=n,brier=b,log_loss=l,delta_brier=b-b0,delta_log_loss=l-l0)
}
calculate <- function(root) {
 rows <- load_records(root);a <- read_chars(file.path(root,"splits/unit09_roles.csv"));check(nrow(a)==nrow(rows) && !anyDuplicated(a$row_id),"manifest uniqueness and coverage")
 for(k in c("row_id","cohort","college_id")) check(identical(as.character(rows[[k]]),a[[k]]),"manifest keys")
 rows$assignment <- a$assignment;check(all(ifelse(rows$college_group=="anchor",rows$assignment=="anchor_reference",ifelse(rows$college_group=="new",rows$assignment %in% c("new_updating","new_final"),rows$assignment=="continuing_final"))),"invalid population role")
 old <- read_chars(file.path(root,"splits/unit08_roles.csv"));check(nrow(old)==nrow(a) && identical(old$row_id,a$row_id) && all(a$assignment==ifelse(old$cohort=="anchor","anchor_reference",old$assignment)),"reused role assignment altered")
 risk <- rows[rows$cohort=="cohort2" & !is.na(rows$y)&!is.na(rows$wellbeing_score)&!is.na(rows$stress_score),,drop=FALSE]
 co <- read_chars(file.path(root,"expected-results/unit07_coefficients.csv"));co <- co[co$model_id=="full_logistic",];tr <- read_chars(file.path(root,"expected-results/unit07_transforms.csv"));tr <- tr[tr$model_id=="full_logistic",];processing <- lapply(seq_len(nrow(tr)),function(j) c(as.numeric(tr$mean[j]),as.numeric(tr$sd[j])));names(processing) <- tr$feature
 risk$p <- plogis(as.numeric(matrix_for(risk,"full_logistic",processing,TRUE) %*% as.numeric(co$coefficient)));risk$eta <- logits(risk$p)
 result <- setNames(rep(list(NULL),10),c("predictions","scores","coefficients","diagnostics","fit_rows","transforms","resampling","resample_summary","profiles","summary"));append <- function(k,frame) result[[k]] <<- rbind(result[[k]],frame)
 register <- function(id,scope,members,terms,fit) {append("coefficients",data.frame(fit_id=id,scope=scope,term=terms,coefficient=fit$b));append("diagnostics",data.frame(fit_id=id,scope=scope,fit$d));append("fit_rows",data.frame(fit_id=id,scope=scope,row_id=members$row_id))}
 update <- risk[risk$assignment=="new_updating",];final <- risk[risk$assignment=="new_final",];sites <- sort(unique(final$college_id));check(length(sites)==4,"four sites");methods <- c("unchanged","offset","joint","refit");fits <- list()
 for(kind in c("offset","joint")) {f <- fit_cal(update$eta,update$y,kind);fits[[kind]] <- f$b;register(paste0("update_",kind),"updating_rule",update,if(kind=="offset") "intercept" else c("intercept","slope"),f)}
 keys <- c("age","prior_grade","wellbeing_score","stress_score");transforms <- setNames(lapply(keys,function(k) {v <- update[[k]];m <- mean(v);sd <- sqrt(mean((v-m)^2));check(is.finite(sd)&&sd>0,"zero SD");c(m,sd)}),keys)
 for(k in keys) append("transforms",data.frame(fit_id="update_refit",feature=k,n=nrow(update),mean=transforms[[k]][1],sd=transforms[[k]][2]))
 design <- function(a) matrix_for(a,"full_logistic",transforms,TRUE)[,1:13,drop=FALSE]
 x <- design(update);f <- fit_general(x,update$y);fits$refit <- f$b;register("update_refit","updating_rule",update,colnames(x),f);newp <- plogis(as.numeric(design(final)%*%f$b));site_scores <- setNames(rep(list(list()),4),sites)
 for(role in c("continuing_final","new_final")) {
  test <- risk[risk$assignment==role,]
  for(method in if(role=="continuing_final") "unchanged" else methods) {
   p <- if(method=="unchanged") test$p else if(method=="refit") newp else plogis(fits[[method]][1]+(if(method=="offset") 1 else fits[[method]][2])*test$eta)
   append("predictions",data.frame(role=role,method=method,college_id=test$college_id,row_id=test$row_id,y=test$y,probability=p))
   for(site in c("all",if(role=="new_final") sites else character())) {
    ids <- if(site=="all") seq_len(nrow(test)) else which(test$college_id==site);sub <- test[ids,];sc <- score(sub$y,p[ids]);append("scores",data.frame(role=role,method=method,college_id=site,sc$s));if(site!="all") site_scores[[site]][[method]] <- sc$s
    for(kind in c("offset","joint")) if(!is.null(sc$cal[[kind]]$fit)) register(paste0("diagnostic_",role,"_",method,"_",site,"_",kind),"diagnostic_only",sub,if(kind=="offset") "intercept" else c("intercept","slope"),sc$cal[[kind]]$fit)
   }
  }
 }
 for(method in methods) for(metric in c("brier","log_loss")) append("summary",data.frame(kind="equal_college_loss",role="new_final",method=method,metric=metric,value=mean(vapply(sites,function(k) site_scores[[k]][[method]][[metric]],numeric(1)))))
 draws <- read_chars(file.path(root,"splits/unit09_college_draws.csv"));check(nrow(draws)==1024,"draw inventory");expected <- list();i <- 0L
 for(a in sites) for(b in sites) for(c in sites) for(d in sites) {i <- i+1L;expected[[i]] <- data.frame(draw_id=as.character(i),position=as.character(1:4),college_id=c(a,b,c,d))}
 check(identical(draws,do.call(rbind,expected)),"whole-college enumeration")
 for(i in 1:256) for(method in methods) append("resampling",data.frame(draw_id=i,method=method,composition(draws$college_id[(4*i-3):(4*i)],site_scores,method)))
 for(method in methods) for(metric in c("brier","log_loss","delta_brier","delta_log_loss")) {
  v <- result$resampling[result$resampling$method==method,metric];qs <- quantile(v,c(.025,.5,.975),type=7,names=FALSE);append("resample_summary",data.frame(method=method,metric=metric,draws=256,mean=mean(v),minimum=min(v),q025=qs[1],median=qs[2],q975=qs[3],maximum=max(v),positive_draws=sum(v>0)))
 }
 for(cohort in c("anchor","cohort2")) {
  frame <- read_chars(file.path(root,"source/tonmick-v1.0.0/data",cohort,"participants.csv"))
  for(group in if(cohort=="anchor") "anchor" else c("continuing","new")) {
   pool <- if(cohort=="anchor") frame else frame[frame$college_group==group,]
   for(response in c("invited","responded")) {
    a <- if(response=="invited") pool else pool[pool$responded_baseline=="1",];append("profiles",data.frame(cohort=cohort,college_group=group,frame_role=response,n=nrow(a),colleges=length(unique(a$college_id)),part_time_fraction=mean(as.integer(a$part_time)),first_gen_fraction=mean(as.integer(a$first_gen)),prior_grade_mean=mean(as.numeric(a$prior_grade))))
   }
  }
 }
 for(role in c("anchor_reference","continuing_final","new_updating","new_final")) {
  a <- rows[rows$assignment==role,];e <- a[!is.na(a$y)&!is.na(a$wellbeing_score)&!is.na(a$stress_score),];v <- c(pool_n=nrow(a),known_scorable_n=nrow(e),events=sum(e$y),unknown_n=sum(is.na(a$y)),colleges=length(unique(a$college_id)))
  for(metric in names(v)) append("summary",data.frame(kind="eligibility",role=role,method="",metric=metric,value=v[[metric]]))
 }
 result
}
args <- commandArgs(trailingOnly=TRUE)
if(length(args)) {
 check(length(args)==2,"usage root outdir");root <- args[1];out <- args[2]
 digest <- strsplit(system2("sha256sum",shQuote(file.path(root,"model-specs/unit09.json")),stdout=TRUE)," ")[[1]][1];check(identical(digest,"cc8810831f589de0358d20d680f2d352f5232c6ccd24f01d6b06bf93a2674d62"),"frozen specification altered")
 options(digits=17);result <- calculate(root);dir.create(out,recursive=TRUE,showWarnings=FALSE)
 for(key in names(result)) {frame <- result[[key]];for(j in seq_along(frame)) if(is.numeric(frame[[j]])) frame[[j]] <- ifelse(is.na(frame[[j]]),"",sprintf("%.17g",frame[[j]]));write.table(frame,file.path(out,paste0(key,".csv")),sep=",",quote=FALSE,row.names=FALSE,na="")}
 cat("UNIT09_BASE_R_OK predictions=",nrow(result$predictions)," college_draws=256 R=",as.character(getRversion()),"\n",sep="")
}
