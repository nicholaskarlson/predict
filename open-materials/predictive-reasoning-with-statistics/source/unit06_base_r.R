#!/usr/bin/env Rscript
# Independent base-R joins, QR ridge, centered coordinate lasso and scores.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)!=2L)stop('usage: unit06_base_r.R RESOURCE OUTDIR')
root <- normalizePath(args[1],mustWork=TRUE); dest <- args[2]
options(digits=17);invisible(Sys.setlocale('LC_COLLATE','C'))
need <- function(ok,message)if(!isTRUE(ok))stop(message)
read <- function(path)read.csv(path,colClasses='character',na.strings=NULL,check.names=FALSE)
raw <- file.path(root,'source/tonmick-v1.0.0/data/anchor');p <- read(file.path(raw,'participants.csv'));b <- read(file.path(raw,'baseline.csv'));y <- read(file.path(raw,'registry.csv'))
for(t in list(p,b,y))need(!anyDuplicated(t$student_id),'duplicate ID')
need(setequal(b$student_id,y$student_id) && setequal(b$student_id,p$student_id[p$responded_baseline=='1']),'respondent join')
ids <- sort(b$student_id);p <- p[match(ids,p$student_id),];b <- b[match(ids,b$student_id),];y <- y[match(ids,y$student_id),];need(all(p$college_id==b$college_id),'college mismatch');keep <- b$offer=='0';p <- p[keep,];y <- y[keep,];pool <- nrow(p);missing <- sum(y$gpa12=='');need(all(y$exit_type!='consent_withdrawn' | y$gpa12==''),'consent grade');keep <- y$gpa12!='';p <- p[keep,];y <- y[keep,]
grade <- as.numeric(y$gpa12);need(all(is.finite(grade)&grade>=0&grade<=4),'invalid grade');need(all(y$gpa_at_exit %in% c('0','1')) && all(y$gpa_at_exit=='0' | as.numeric(y$exit_month)>=4),'exit grade timing');age <- as.numeric(p$age);prior <- as.numeric(p$prior_grade);need(all(p$part_time %in% c('0','1')) && all(p$first_gen %in% c('0','1')),'binary predictor');need(all(is.finite(age)&age>=16&age<=80) && all(is.finite(prior)&prior>=0&prior<=4),'entry predictor')
people <- data.frame(row_id=p$student_id,college_id=p$college_id,region=p$region,part_time=as.numeric(p$part_time),first_gen=as.numeric(p$first_gen),age=age,prior_grade=prior,y=grade,gpa_at_exit=as.integer(y$gpa_at_exit),role=ifelse(p$region=='West','development','setting_check'),stringsAsFactors=FALSE)
dev <- people[people$role=='development',];setting <- people[people$role=='setting_check',];outer <- read(file.path(root,'splits/unit06_outer.csv'));inner <- read(file.path(root,'splits/unit06_inner.csv'));need(nrow(outer)==nrow(dev) && !anyDuplicated(outer$row_id) && setequal(outer$row_id,dev$row_id),'outer membership');outer <- outer[match(dev$row_id,outer$row_id),];need(all(outer$college_id==dev$college_id) && all(outer$outer_fold %in% as.character(1:5)) && setequal(outer$outer_fold,as.character(1:5)),'outer college/fold');contexts <- c('full',paste0('outer',1:5));need(all(inner$context %in% contexts),'unexpected context');nested <- list()
for(context in contexts){
 train <- dev[context=='full' | outer$outer_fold!=substr(context,6,6),];z <- inner[inner$context==context,];need(nrow(z)==nrow(train) && !anyDuplicated(z$row_id) && setequal(z$row_id,train$row_id),'inner membership');z <- z[match(train$row_id,z$row_id),];need(all(z$college_id==train$college_id) && all(z$inner_fold %in% as.character(1:5)) && setequal(z$inner_fold,as.character(1:5)),'inner college/fold');nested[[context]] <- setNames(z$inner_fold,z$row_id)
}
# Base R only: a deliberately small parser for the frozen string arrays.
spec <- paste(readLines(file.path(root,'model-specs/unit06.json')),collapse='\n')
array <- function(key){block <- regmatches(spec,regexpr(paste0('"',key,'"[[:space:]]*:[[:space:]]*\\[[^]]*\\]'),spec,perl=TRUE));tokens <- regmatches(block,gregexpr('"[^"]+"',block))[[1]];gsub('"','',tokens[-1],fixed=TRUE)}
features <- c('prior_grade','part_time','first_gen','age','prior_grade_squared','age_squared','prior_grade_part_time','prior_grade_first_gen','prior_grade_age');models <- c('entry','rich','ridge','lasso');grids <- list(ridge=c('0','0.1','1','10','100','1000'),lasso=c('0','0.1','0.3','1','3','10','30','100'))
need(identical(array('predictor_allowlist'),c('prior_grade','part_time','first_gen','age')) && identical(array('rich_features'),features) && identical(array('model_ids'),models),'model spec')
for(k in names(grids))need(identical(array(paste0(k,'_grid')),grids[[k]]),'penalty grid')
need(grepl('"setting_labels_for_selection": false',spec,fixed=TRUE) && grepl('"intercept_penalized": false',spec,fixed=TRUE) && grepl('"lambda_scaling": "fixed_sum_scale_no_fold_n_rescaling"',spec,fixed=TRUE),'selection contract')
outputs <- setNames(lapply(1:9,function(i)list()),c('predictions','coefficients','diagnostics','transforms','fit_rows','tuning','selection','scores','summary'))
add <- function(name,rows)outputs[[name]][[length(outputs[[name]])+1L]] <<- rows
terms <- function(t,model){g <- t$prior_grade;a <- t$age;if(model=='entry')matrix(g,ncol=1) else cbind(g,t$part_time,t$first_gen,a,g*g,a*a,g*t$part_time,g*t$first_gen,g*a)}
processing <- function(t,model){x <- terms(t,model);mu <- colMeans(x);sd <- sqrt(colMeans(sweep(x,2,mu,'-')^2));need(all(is.finite(sd)&sd>1e-12),'zero SD');list(mean=mu,sd=sd)}
matrix_for <- function(t,model,parameters)cbind(1,sweep(sweep(terms(t,model),2,parameters$mean,'-'),2,parameters$sd,'/'))
qrsolve <- function(x,y){q <- qr(x,tol=1e-10,LAPACK=FALSE);r <- qr.R(q);d <- abs(diag(r));need(q$rank==ncol(x) && min(d)>1e-10*max(d),'QR rank');condition <- norm(r,'I')*norm(backsolve(r,diag(ncol(r))),'I');need(is.finite(condition) && condition<=1e5,'QR condition');list(beta=as.numeric(qr.coef(q,y)),condition=condition)}
fit <- function(x,y,method,lam,max_sweeps=100000L){
 n <- length(y);p <- ncol(x);initial <- qrsolve(x,y);beta <- initial$beta;iterations <- 0L
 if(method=='ridge' && lam>0){extra <- cbind(0,sqrt(lam)*diag(p-1));beta <- qrsolve(rbind(x,extra),c(y,rep(0,p-1)))$beta}
 if(method=='lasso' && lam>0){
  means <- colMeans(x[,-1,drop=FALSE]);z <- sweep(x[,-1,drop=FALSE],2,means,'-');ym <- mean(y);gram <- crossprod(z);cross <- as.numeric(crossprod(z,y-ym));slopes <- rep(0,p-1);converged <- FALSE
  for(iterations in seq_len(max_sweeps)){
   change <- 0
   for(j in seq_len(p-1)){old <- slopes[j];partial <- cross[j]-sum(gram[j,-j]*slopes[-j]);slopes[j] <- sign(partial)*max(0,abs(partial)-lam/2)/gram[j,j];change <- max(change,abs(slopes[j]-old))}
   gradient <- as.numeric(2*(gram %*% slopes-cross));violation <- ifelse(slopes!=0,abs(gradient+lam*sign(slopes)),pmax(abs(gradient)-lam,0));kkt <- max(violation)/n
   if(change<=1e-12 && kkt<=1e-10){converged <- TRUE;break}
  }
  need(converged,'lasso did not converge');beta <- c(ym-sum(means*slopes),slopes)
 }
 residual <- as.numeric(x %*% beta-y);gradient <- as.numeric(2*crossprod(x,residual))
 if(method=='ridge')gradient[-1] <- gradient[-1]+2*lam*beta[-1]
 if(method=='lasso')gradient[-1] <- ifelse(beta[-1]!=0,abs(gradient[-1]+lam*sign(beta[-1])),pmax(abs(gradient[-1])-lam,0))
 kkt <- max(abs(gradient))/n;need(kkt<=1e-10,'fit KKT');rss <- sum(residual^2);penalty <- if(method=='ridge')lam*sum(beta[-1]^2) else if(method=='lasso')lam*sum(abs(beta[-1])) else 0
 list(beta=beta,diagnostic=data.frame(n=n,p=p,rank=p,condition_r_inf=initial$condition,iterations=iterations,rss=rss,penalty=penalty,objective=rss+penalty,kkt=kkt,nonzero=sum(abs(beta[-1])>1e-10)))
}
score <- function(y,p){need(length(y)>0 && all(is.finite(p)),'score');sst <- sum((p-mean(p))^2);slope <- if(sst>1e-12)sum((p-mean(p))*(y-mean(y)))/sst else '';intercept <- if(sst>1e-12)mean(y)-as.numeric(slope)*mean(p) else '';data.frame(n=length(y),mse=sum((y-p)^2)/length(y),mae=sum(abs(y-p))/length(y),calibration_intercept=intercept,calibration_slope=slope,calibration_reason=if(sst>1e-12)'' else 'constant_prediction',outside_0_4=sum(p<0|p>4))}
# Orthogonal, centered design: ridge and lasso answers known by hand.
x0 <- cbind(1,c(-1,0,1));y0 <- c(0,1,2);need(max(abs(fit(x0,y0,'ridge',2)$beta-c(1,.5)))<1e-12,'ridge oracle');need(max(abs(fit(x0,y0,'lasso',2)$beta-c(1,.5)))<1e-12,'lasso oracle');need(max(abs(fit(x0,y0,'lasso',4)$beta-c(1,0)))<1e-12,'lasso zero oracle')
need(inherits(try(fit(x0,y0,'lasso',2,max_sweeps=1L),silent=TRUE),'try-error'),'convergence failure oracle')
fitted <- function(context,stage,model,lam,train,evaluations){
 fid <- paste(context,stage,model,lam,sep='|');parameters <- processing(train,model);x <- matrix_for(train,model,parameters);result <- fit(x,train$y,model,as.numeric(lam));columns <- if(model=='entry')'prior_grade' else features
 add('diagnostics',cbind(data.frame(fit_id=fid,context=context,stage=stage,model_id=model,lambda_value=lam),result$diagnostic));add('coefficients',data.frame(fit_id=fid,term=c('intercept',columns),coefficient=result$beta));add('transforms',data.frame(fit_id=fid,column=columns,mean=parameters$mean,sd=parameters$sd));add('fit_rows',data.frame(fit_id=fid,row_id=train$row_id))
 for(role in names(evaluations)){held <- evaluations[[role]];prediction <- as.numeric(matrix_for(held,model,parameters) %*% result$beta);add('predictions',data.frame(fit_id=fid,context=context,role=role,model_id=model,lambda_value=lam,row_id=held$row_id,y=held$y,prediction=prediction))}
}
for(context in contexts){
 train <- dev[context=='full' | outer$outer_fold!=substr(context,6,6),];selected <- list()
 for(model in c('ridge','lasso')){
  losses <- numeric(length(grids[[model]]));names(losses) <- grids[[model]]
  for(lam in grids[[model]]){
   start <- length(outputs$predictions)
   for(j in as.character(1:5)){fold <- nested[[context]][train$row_id];fitted(context,paste0('inner',j),model,lam,train[fold!=j,],list(inner_validation=train[fold==j,]))}
   pp <- do.call(rbind,outputs$predictions[(start+1):length(outputs$predictions)]);need(nrow(pp)==nrow(train),'inner prediction coverage');loss <- sum((pp$y-pp$prediction)^2)/nrow(pp);losses[lam] <- loss;add('tuning',data.frame(context=context,model_id=model,lambda_value=lam,n=nrow(pp),mse=loss))
  }
  minimum <- min(losses);eligible <- names(losses)[losses<=minimum+1e-10];chosen <- eligible[which.max(as.numeric(eligible))];selected[[model]] <- chosen;add('selection',data.frame(context=context,model_id=model,lambda_value=chosen,minimum_mse=minimum,selected_mse=unname(losses[chosen])))
 }
 evaluations <- if(context=='full')list(apparent=train,setting_check=setting) else list(outer_validation=dev[outer$outer_fold==substr(context,6,6),])
 for(model in models)fitted(context,'refit',model,if(model %in% names(selected))selected[[model]] else '0',train,evaluations)
}
predictions <- do.call(rbind,outputs$predictions)
for(scope in c('full_apparent','full_setting_check',paste0('outer',1:5),'pooled')){
 context <- if(scope %in% c('full_apparent','full_setting_check'))'full' else scope;role <- if(scope=='full_apparent')'apparent' else if(scope=='full_setting_check')'setting_check' else 'outer_validation'
 for(model in models){pp <- predictions[predictions$role==role & predictions$model_id==model & (context=='pooled'|predictions$context==context),];add('scores',cbind(data.frame(context=context,role=role,model_id=model),score(pp$y,pp$prediction)))}
}
counts <- c(pool_n=pool,missing_n=missing,development_n=nrow(dev),setting_n=nrow(setting),development_colleges=length(unique(dev$college_id)),setting_colleges=length(unique(setting$college_id)),development_exit_grades=sum(dev$gpa_at_exit),setting_exit_grades=sum(setting$gpa_at_exit),inner_memberships=nrow(inner),fits=length(outputs$diagnostics),fit_memberships=sum(vapply(outputs$fit_rows,nrow,integer(1))),prediction_records=nrow(predictions))
for(k in sort(names(counts)))add('summary',data.frame(name=k,value=unname(counts[k])))
dir.create(dest,recursive=TRUE,showWarnings=FALSE)
for(k in names(outputs))write.table(do.call(rbind,outputs[[k]]),file.path(dest,paste0(k,'.csv')),sep=',',row.names=FALSE,col.names=TRUE,quote=TRUE,na='')
cat('UNIT06_BASE_R_OK predictions=',nrow(predictions),' R=',as.character(getRversion()),'\n',sep='')
