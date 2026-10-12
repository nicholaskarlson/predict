#!/usr/bin/env Rscript
# Independent base-R path: original observed joins, QR imputation, BFGS/Newton ridge and rank AUC.
methods <- c('frame_only','survey_fallback','imputed','imputed_indicators')
scales <- c('wellbeing_score','stress_score')
indicators <- c('survey_absent','wb_item_missing','st_item_missing')
frame_names <- c('part_time','first_gen','age','gender','program_area','prior_grade')
items <- c(paste0('wb',1:5),paste0('st',1:5))
programs <- c('Arts and Humanities','Social Sciences','STEM','Health','Business')
genders <- c('female','male','another')
patterns <- c('survey_absent','scores_complete_items','scores_partial_wb','scores_partial_st',
              'scores_partial_both','wb_unscorable','st_unscorable','both_unscorable')
terms_frame <- c('intercept','part_time','first_gen','z_age','gender_male','gender_another',
                'program_Social_Sciences','program_STEM','program_Health','program_Business','z_prior_grade')
check <- function(ok,message) { if (!isTRUE(ok)) stop(message,call.=FALSE) }
read_chars <- function(path) read.csv(path,colClasses='character',na.strings=character(),check.names=FALSE,
                                    stringsAsFactors=FALSE,strip.white=FALSE)
unique_rows <- function(path,key='student_id') {
  x <- read_chars(path);check(!anyDuplicated(names(x)) && !anyDuplicated(x[[key]]) && all(nzchar(x[[key]])), 'duplicate/empty ID or column');x
}
profiles <- function(x) {
  check(all(x$gender %in% genders) && all(x$program_area %in% programs),'unsupported frame category')
  for (name in c('part_time','first_gen','responded_baseline')) {
    check(all(as.character(x[[name]]) %in% c('0','1')),'invalid binary value');x[[name]] <- as.integer(x[[name]])
  }
  for (key in c('age','prior_grade')) x[[key]] <- as.numeric(x[[key]])
  check(all(is.finite(x$age)&x$age>=16&x$age<=80) && all(is.finite(x$prior_grade)&x$prior_grade>=0&x$prior_grade<=4),'invalid frame value')
  for (j in seq_along(scales)) {
    prefix <- c('wb','st')[j]; a <- matrix(NA_real_,nrow(x),5)
    for (i in 1:5) {
      name <- paste0(prefix,i); v <- x[[name]]
      if (is.null(v)) v <- rep('',nrow(x))
      present <- !is.na(v)&v!='';check(all(as.character(v[present]) %in% as.character(1:5)), 'invalid scale item')
      value <- rep(NA_integer_,nrow(x));value[present] <- as.integer(v[present]);x[[name]] <- value
      a[,i] <- value;if (i==4) a[,i] <- 6-a[,i]
    }
    count <- rowSums(!is.na(a));score <- rowSums(a,na.rm=TRUE)/count;score[count<3] <- NA_real_
    x[[paste0(prefix,'_nobs')]] <- count;x[[scales[j]]] <- score
  }
  check(all(x$responded_baseline==1 | x$wb_nobs+x$st_nobs==0),'nonresponse/items conflict')
  x$survey_absent <- 1-x$responded_baseline;x$wb_item_missing <- as.integer(x$wb_nobs<5);x$st_item_missing <- as.integer(x$st_nobs<5)
  x$pattern <- 'scores_complete_items'
  x$pattern[x$wb_item_missing==1] <- 'scores_partial_wb'
  x$pattern[x$st_item_missing==1] <- 'scores_partial_st'
  x$pattern[x$wb_item_missing==1 & x$st_item_missing==1] <- 'scores_partial_both'
  x$pattern[is.na(x$wellbeing_score)] <- 'wb_unscorable'
  x$pattern[is.na(x$stress_score)] <- 'st_unscorable'
  x$pattern[is.na(x$wellbeing_score)&is.na(x$stress_score)] <- 'both_unscorable'
  x$pattern[x$survey_absent==1] <- 'survey_absent';x
}
load_records <- function(root) {
  legacy <- unique_rows(file.path(root,'splits/unit09_roles.csv'),'row_id');pieces <- list()
  for (cohort in c('anchor','cohort2')) {
    folder <- file.path(root,'source/tonmick-v1.0.0/data',cohort)
    f <- unique_rows(file.path(folder,'participants.csv'));s <- unique_rows(file.path(folder,'baseline.csv'))
    o <- unique_rows(file.path(root,'source/tonmick-observed-v1.1.0/data',paste0(cohort,'_observed_registry.csv')))
    check(setequal(f$student_id,o$student_id) && setequal(s$student_id,f$student_id[f$responded_baseline=='1']),'invitation/response join')
    f <- f[order(f$student_id),,drop=FALSE];o <- o[match(f$student_id,o$student_id),,drop=FALSE]
    check(all(o$cohort==cohort & o$responded_baseline==f$responded_baseline),'registry cohort/response join')
    s <- s[match(f$student_id,s$student_id),,drop=FALSE]
    check(all(is.na(s$college_id)|s$college_id==f$college_id),'survey college join')
    x <- data.frame(row_id=f$student_id,cohort=cohort,college_id=f$college_id,region=f$region,
      college_group=if (cohort=='anchor') 'anchor' else f$college_group,f[,frame_names,drop=FALSE],
      responded_baseline=f$responded_baseline,s[,items,drop=FALSE],stringsAsFactors=FALSE)
    x <- profiles(x)
    check(all(o$offer %in% c('0','1') & o$outcome_known %in% c('0','1')),'observation flags')
    offer <- as.integer(o$offer);known <- as.integer(o$outcome_known)
    check(all((o$withdrawal_label=='')==(known==0)),'unknown label must be blank')
    y <- rep(NA_integer_,nrow(x));y[known==1] <- as.integer(o$withdrawal_label[known==1])
    check(all(y[known==1] %in% 0:1 & y[known==1]==as.integer(o$exit_type[known==1]=='withdrawal')),'label/event')
    check(all(known==1|o$exit_type=='consent_withdrawn'),'unknown event')
    check(all(x$survey_absent==0|(offer==0&known==1)),'nonresponse observation policy')
    check(cohort=='anchor'||all(offer==0),'later offer policy')
    if (cohort=='anchor') role <- ifelse(offer==1,'excluded_offered',ifelse(known==1,'development','excluded_unknown')) else {
      check(all(x$college_group %in% c('continuing','new')),'later college group');role <- paste0('temporal_',x$college_group)
    }
    previous <- legacy$assignment[match(x$row_id,legacy$row_id)];previous[is.na(previous)] <- 'not_in_prior_respondent_analysis'
    x$offer <- offer;x$outcome_known <- known;x$y <- y;x$exit_type <- o$exit_type;x$observation_source <- o$observation_source
    x$role <- role;x$legacy_role <- previous;pieces[[cohort]] <- x
  }
  x <- do.call(rbind,pieces);x <- x[order(x$row_id),,drop=FALSE];rownames(x) <- NULL
  check(!anyDuplicated(x$row_id),'cohort ID collision');x
}
processing <- function(data,keys) {
  setNames(lapply(keys,function(key) {
    x <- data[[key]];m <- mean(x);s <- sqrt(mean((x-m)^2));check(all(is.finite(x))&&is.finite(s)&&s>0,'invalid/zero SD');c(m,s)
  }),keys)
}
design <- function(data,tr,with_scales=FALSE,with_indicators=FALSE) {
  z <- function(key) (data[[key]]-tr[[key]][1])/tr[[key]][2]
  x <- cbind(intercept=rep(1,nrow(data)),part_time=data$part_time,first_gen=data$first_gen,z_age=z('age'),
             gender_male=as.integer(data$gender=='male'),gender_another=as.integer(data$gender=='another'))
  for (level in programs[-1]) { x <- cbind(x,as.integer(data$program_area==level));colnames(x)[ncol(x)] <- paste0('program_',gsub(' ','_',level)) }
  x <- cbind(x,z_prior_grade=z('prior_grade'))
  if (with_scales) for (key in scales) { x <- cbind(x,z(key));colnames(x)[ncol(x)] <- paste0('z_',key) }
  if (with_indicators) for (key in indicators) {x <- cbind(x,data[[key]]);colnames(x)[ncol(x)] <- key}
  check(all(is.finite(x)), 'invalid design');x
}
qr_solution <- function(x,y) {
  check(nrow(x)>ncol(x)&&length(y)==nrow(x),'fit dimensions')
  q <- qr(x,tol=1e-10,LAPACK=FALSE);check(q$rank==ncol(x)&&identical(q$pivot,seq_len(ncol(x))),'rank failure')
  triangular <- qr.R(q);d <- abs(diag(triangular));check(min(d)>1e-10*max(d),'rank failure')
  cond <- norm(triangular,'I')*norm(backsolve(triangular,diag(ncol(x))),'I');check(is.finite(cond)&&cond<=1e5,'conditioning failure')
  list(beta=as.numeric(qr.coef(q,y)),condition=cond)
}
fit_logistic <- function(x,y) {
  check(setequal(unique(y),0:1),'training classes')
  penalty <- diag(ncol(x))[-1,,drop=FALSE] # lambda=1, intercept excluded
  objective <- function(beta) {
    eta <- as.numeric(x%*%beta);sign <- ifelse(y==1,-eta,eta)
    sum(pmax(0,sign)+log1p(exp(-abs(sign)))) + sum(beta[-1]^2)/2
  }
  gradient <- function(beta) as.numeric(crossprod(x,plogis(x%*%beta)-y))+c(0,beta[-1])
  start <- c(log(sum(y)/(length(y)-sum(y))),rep(0,ncol(x)-1))
  fit <- optim(start,objective,gradient,method='BFGS',control=list(maxit=100,reltol=1e-12))
  check(fit$convergence==0,'BFGS nonconvergence');beta <- fit$par
  for (iteration in 0:100) {
    eta <- as.numeric(x%*%beta);p <- plogis(eta);w <- p*(1-p)
    check(all(is.finite(w)&w>0),'invalid weight')
    last <- qr_solution(rbind(x*sqrt(w),penalty),c((y-p)/sqrt(w),-beta[-1]))
    g <- -gradient(beta);score <- max(abs(g));step <- max(abs(last$beta));nll <- objective(beta)
    if (score<=1e-9 && step<=1e-9) {
      check(all(is.finite(beta))&&max(abs(beta))<=30,'nonfinite/excess coefficient')
      return(list(beta=beta,diag=data.frame(n_fit=length(y),events=sum(y),parameters=length(beta),rank=length(beta),objective=nll,
        score_inf=score,step_inf=step,condition_r_inf=last$condition,minimum_weight=min(w))))
    }
    check(iteration<100,'Newton nonconvergence');alpha <- 1;accepted <- FALSE
    for (attempt in 1:40) {
      candidate <- beta+alpha*last$beta
      if (objective(candidate)<=nll-1e-4*alpha*sum(g*last$beta)+1e-12) {beta <- candidate;accepted <- TRUE;break}
      alpha <- alpha/2
    };check(accepted,'Newton backtracking failed')
  };stop('Newton nonconvergence')
}
complete_data <- function(data,imputers) {
  x <- data
  for (key in scales) {
    missing <- is.na(x[[key]]);m <- imputers[[key]]
    if (any(missing)) {
      z <- design(x[missing,,drop=FALSE],m$transforms);check(identical(colnames(z),m$terms),'imputer term order')
      x[[key]][missing] <- pmax(1,pmin(5,as.numeric(z%*%m$coefficients)))
    }
    x[[if (key=='wellbeing_score') 'wb_imputed' else 'st_imputed']] <- as.integer(missing)
  };x
}
fit_pipelines <- function(rows) {
  train <- rows[rows$role=='development',,drop=FALSE]
  check(nrow(train)==984 && all(train$cohort=='anchor' & train$offer==0 & train$y %in% 0:1),'training membership')
  output <- list(coefficients=list(),transforms=list(),diagnostics=list(),fit_rows=list())
  store <- function(name,tr,beta,terms,data,diag) {
    output$coefficients[[length(output$coefficients)+1L]] <<- data.frame(component=name,term=terms,coefficient=beta)
    output$transforms[[length(output$transforms)+1L]] <<- do.call(rbind,lapply(names(tr),function(k) data.frame(component=name,feature=k,n_fit=nrow(data),mean=tr[[k]][1],sd=tr[[k]][2])))
    output$diagnostics[[length(output$diagnostics)+1L]] <<- data.frame(component=name,diag)
    output$fit_rows[[length(output$fit_rows)+1L]] <<- data.frame(component=name,row_id=data$row_id)
  }
  imputers <- list()
  for (key in scales) {
    data <- train[!is.na(train[[key]]),,drop=FALSE];tr <- processing(data,c('age','prior_grade'));x <- design(data,tr)
    f <- qr_solution(x,data[[key]]);residual <- data[[key]]-as.numeric(x%*%f$beta)
    diag <- data.frame(n_fit=nrow(x),events='',parameters=ncol(x),rank=ncol(x),objective=sum(residual^2),
      score_inf=max(abs(as.numeric(crossprod(x,residual)))),step_inf='',condition_r_inf=f$condition,minimum_weight='')
    store(paste0('imputer_',key),tr,f$beta,colnames(x),data,diag)
    imputers[[key]] <- list(transforms=tr,terms=colnames(x),coefficients=f$beta)
  }
  filled <- complete_data(train,imputers);models <- list()
  for (method in methods) {
    with_scales <- method!='frame_only';with_indicators <- method=='imputed_indicators'
    data <- if (method=='survey_fallback') train[!is.na(train$wellbeing_score)&!is.na(train$stress_score),,drop=FALSE] else if (method %in% c('imputed','imputed_indicators')) filled else train
    tr <- processing(data,c('age','prior_grade',if (with_scales) scales else character()));x <- design(data,tr,with_scales,with_indicators)
    f <- fit_logistic(x,data$y);store(method,tr,f$beta,colnames(x),data,f$diag)
    models[[method]] <- list(transforms=tr,terms=colnames(x),coefficients=f$beta)
  }
  counts <- setNames(vapply(patterns,function(p) sum(train$pattern==p),integer(1)),patterns)
  list(bundle=list(imputers=imputers,models=models,training_pattern_counts=counts),output=lapply(output,function(x) do.call(rbind,x)))
}
predict_profiles <- function(data,bundle) {
  original <- profiles(data);filled <- complete_data(original,bundle$imputers);result <- list()
  for (method in methods) {
    route <- rep(method,nrow(data))
    if (method=='survey_fallback') route[is.na(original$wellbeing_score)|is.na(original$stress_score)] <- 'frame_only'
    r <- if (method %in% c('imputed','imputed_indicators')) filled else original
    eta <- rep(NA_real_,nrow(data))
    for (name in unique(route)) {
      chosen <- route==name;m <- bundle$models[[name]]
      x <- design(r[chosen,,drop=FALSE],m$transforms,name!='frame_only',name=='imputed_indicators')
      check(identical(colnames(x),m$terms),'prediction term order');eta[chosen] <- as.numeric(x%*%m$coefficients)
    }
    n <- unname(bundle$training_pattern_counts[original$pattern])
    result[[method]] <- data.frame(method_id=method,record_id=if ('record_id' %in% names(data)) data$record_id else data$row_id,
      route=route,pattern=original$pattern,training_pattern_n=n,support_reason=ifelse(n>0,'','unseen_missingness_pattern'),
      wellbeing_score=filled$wellbeing_score,stress_score=filled$stress_score,wb_imputed=filled$wb_imputed,st_imputed=filled$st_imputed,
      eta=eta,probability=plogis(eta),stringsAsFactors=FALSE)
  }
  out <- do.call(rbind,result);out <- out[order(rep(seq_len(nrow(data)),length(methods)),rep(seq_along(methods),each=nrow(data))),,drop=FALSE]
  rownames(out) <- NULL;out
}
score <- function(y,p,n_predict) {
  check(length(y)==length(p)&&all(y %in% 0:1),'score labels');check(all(is.finite(p)&p>=0&p<=1),'score probabilities')
  n <- length(y);e <- sum(y);ne <- n-e;den <- 2*e*ne
  twice <- if (den) 2*(sum(rank(p,ties.method='average')[y==1])-e*(e+1)/2) else NA_real_
  data.frame(n_predict=n_predict,n_scored=n,unknown_n=n_predict-n,events=e,nonevents=ne,auc_pairs2=twice,auc_denominator=den,
    auc=if (den) twice/den else NA_real_,auc_reason=if (den) '' else if (!n) 'empty' else 'one_class',
    brier=if (n) mean((p-y)^2) else NA_real_,mean_probability=if (n) mean(p) else NA_real_,stringsAsFactors=FALSE)
}
calculate <- function(root) {
  cfg <- read_chars(file.path(root,'model-specs/unit11_contract.csv'));check(identical(cfg$key,c('min_items','reverse_item','reverse_constant','clip_low','clip_high','maxit','score_limit','step_limit','coefficient_limit','rank_limit','condition_limit','ridge_lambda')) &&
    all(as.numeric(cfg$value)==c(3,4,6,1,5,100,1e-9,1e-9,30,1e-10,1e5,1)),'base-R contract mirror')
  md <- read_chars(file.path(root,'model-specs/unit11_models.csv'));check(identical(md$method_id,methods),'method order')
  rows <- load_records(root);fit <- fit_pipelines(rows);bundle <- fit$bundle;out <- fit$output
  later <- rows[rows$cohort=='cohort2',,drop=FALSE];p <- predict_profiles(later,bundle)
  all <- complete_data(rows,bundle$imputers)
  out$completed <- all[,c('row_id','role','pattern','wellbeing_score','stress_score','wb_imputed','st_imputed'),drop=FALSE]
  predictions <- list();scores <- list();coverage <- list()
  for (method in methods) {
    pm <- p[p$method_id==method,,drop=FALSE];check(identical(pm$record_id,later$row_id),'prediction IDs')
    predictions[[length(predictions)+1L]] <- data.frame(method_id=method,row_id=later$row_id,role=later$role,
      college_id=later$college_id,legacy_role=later$legacy_role,responded_baseline=later$responded_baseline,pattern=later$pattern,
      y=later$y,outcome_known=later$outcome_known,route=pm$route,probability=pm$probability,eta=pm$eta)
    for (role in c('temporal_continuing','temporal_new')) {
      chosen <- later$role==role;population <- later[chosen,,drop=FALSE];pr <- pm[chosen,,drop=FALSE]
      for (pat in c('all',patterns)) {
        members <- if (pat=='all') rep(TRUE,nrow(population)) else population$pattern==pat
        known <- members & population$outcome_known==1
        scores[[length(scores)+1L]] <- data.frame(method_id=method,role=role,pattern=pat,
          score(population$y[known],pr$probability[known],sum(members)))
      }
      coverage[[length(coverage)+1L]] <- data.frame(method_id=method,role=role,invited=nrow(population),predictions=nrow(population),
        labels_known=sum(population$outcome_known),labels_unknown=sum(population$outcome_known==0),
        survey_route_n=sum(pr$route=='survey_fallback'),frame_fallback_n=sum(method=='survey_fallback' & pr$route=='frame_only'),
        nonrespondents=sum(population$survey_absent),partial_item_respondents=sum(population$survey_absent==0 & (population$wb_item_missing==1 | population$st_item_missing==1)))
    }
  }
  out$predictions <- do.call(rbind,predictions);out$scores <- do.call(rbind,scores);out$coverage <- do.call(rbind,coverage)
  out$new_predictions <- predict_profiles(read_chars(file.path(root,'deployment/unit11_new_profiles.csv')),bundle)
  summary <- c(anchor_invited=1600,anchor_offered_excluded=sum(rows$role=='excluded_offered'),anchor_unknown_no_offer_excluded=sum(rows$role=='excluded_unknown'),
    development_n=sum(rows$role=='development'),development_respondents=sum(rows$role=='development' & rows$responded_baseline==1),
    development_nonrespondents=sum(rows$role=='development' & rows$survey_absent==1),later_invited=nrow(later),later_nonrespondents=sum(later$survey_absent),
    later_labels_known=sum(later$outcome_known),training_unscorable_respondents=sum(rows$role=='development' & rows$survey_absent==0 & (is.na(rows$wellbeing_score)|is.na(rows$stress_score))),new_world_random_draws=0)
  out$summary <- data.frame(name=sort(names(summary)),value=unname(summary[sort(names(summary))]));out
}
emit <- function(output,folder) {
  dir.create(folder,recursive=TRUE,showWarnings=FALSE)
  for (name in names(output)) {
    d <- output[[name]]
    for (key in names(d)) if (is.numeric(d[[key]])) d[[key]] <- vapply(d[[key]],function(x) if (is.na(x)) '' else sprintf('%.17g',x),character(1))
    write.table(d,file.path(folder,paste0(name,'.csv')),sep=',',row.names=FALSE,col.names=TRUE,quote=FALSE,na='')
  }
  writeLines(capture.output(sessionInfo()),file.path(folder,'session-info.txt'))
}
if (sys.nframe()==0L) {
  args <- commandArgs(trailingOnly=TRUE);check(length(args)==2L,'usage: Rscript unit11_base_r.R RESOURCE OUTDIR')
  Sys.setlocale('LC_COLLATE','C');options(digits=17)
  emit(calculate(args[1]),args[2]);cat('UNIT11_BASE_R_OK methods=4 imputers=2 later_predictions=14368 outcome_free_new_profiles=3 R=',as.character(getRversion()),'\n',sep='')
}
